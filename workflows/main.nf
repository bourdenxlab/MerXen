nextflow.enable.dsl = 2

include { BUILD_SPATIALDATA } from "./modules/spatialdata_build"
include { ENSURE_PROSEG } from "./modules/proseg_bootstrap"
include { CELLPOSE_NUCLEI_SEGMENT; SEGMENT } from "./modules/segmentation"
include { NATIVE_LATEST } from "./modules/native_segmentation"
include { ATERA_PROSEG_PLAN; ATERA_PROSEG_PLAN_CELLPOSE; ATERA_PROSEG_TILE; ATERA_PROSEG_STITCH } from "./modules/atera_proseg"
include { ATERA_CELLPOSE_SEGMENT; ATERA_STOCK_CELLPOSE_SEGMENT } from "./modules/atera_cellpose"
include { ATERA_TRANSCRIPT_DENSITY } from "./modules/atera_transcript_density"
include { CELLPOSE_HITL_TRAIN; CELLPOSE_HITL_SEED } from "./modules/cellpose_hitl"
include { ENRICH } from "./modules/enrichment"
include { VIEWER_CACHE } from "./modules/viewer_cache"
include { MASK_IMAGE_QUANTIFICATION } from "./modules/mask_image_quantification"
include { REGISTER_IMAGES } from "./modules/register_images"
include { COMPUTE_CORTICAL_DEPTH } from "./modules/compute_cortical_depth"
include {
    DISTANCE_FROM_OBJECT_ANNOTATE;
    DISTANCE_FROM_OBJECT_COHORT
} from "./modules/distance_from_object"
include { VALIDATE_ANALYSIS_LAYER; QC } from "./modules/qc"
include { ALIGN; MATERIALIZE_ALIGNMENT; ALIGN_QC } from "./modules/alignment"
include { COMPARE } from "./modules/comparison"
include { VISUALIZE } from "./modules/visualization"
include { SPATIAL_GENE_ANALYSIS } from "./modules/spatial_gene_analysis"
include { MECR_REFERENCE; MECR_REFERENCE_ATERA; MECR } from "./modules/mecr"
include {
    CLUSTERING_SQUIDPY_PREPARE;
    CLUSTERING_SQUIDPY_COMPUTE;
    CLUSTERING_SQUIDPY_FINALIZE
} from "./modules/clustering_squidpy"
include {
    PCISEQ_REFERENCE;
    PCISEQ_PREPARE;
    PCISEQ_COMPUTE;
    PCISEQ_FINALIZE
} from "./modules/pciseq"
include { MAPMYCELLS } from "./modules/mapmycells"
include {
    MENDER_PREPARE;
    MENDER_COMPUTE;
    MENDER_FINALIZE;
    MENDER_IMPORT
} from "./modules/mender"
include {
    GASTON_PREPARE;
    GASTON_GLM_PCA;
    GASTON_TRAIN;
    GASTON_POSTPROCESS;
    GASTON_IMPORT
} from "./modules/gaston"
// rca-hook:H1: reference-based annotation (plan §2.4); nothing runs in legacy mode.
include { ANNOTATION_PREPARE_ONLY } from "./subworkflows/annotation_references"
include { CLUSTERING_MAP_FIRST } from "./subworkflows/clustering_map_first"
include { HIPPOCAMPAL_ANNOTATION; HIPPOCAMPAL_REPORT; HIPPOCAMPAL_REFERENCE as HIPPOCAMPAL_PREPARE_ONLY } from "./subworkflows/hippocampal_annotation"
include { ANNOTATION_REPORTING } from "./subworkflows/annotation_report"

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
        "atera": "atera",
        "a": "atera",
    ]
    if (!aliases.containsKey(key)) {
        throw new IllegalArgumentException(
            "Unknown analysis_mode '${raw}'. Valid values: paired, merscope, xenium"
        )
    }
    return aliases[key]
}

def normalizeSpecies(rawValue) {
    def raw = rawValue == null ? "human" : rawValue.toString().trim().toLowerCase()
    def aliases = [
        "human": "human",
        "homo_sapiens": "human",
        "homo sapiens": "human",
        "mouse": "mouse",
        "mus_musculus": "mouse",
        "mus musculus": "mouse",
    ]
    if (!aliases.containsKey(raw)) {
        throw new IllegalArgumentException(
            "Unknown species '${rawValue}'. Valid values: human, mouse"
        )
    }
    return aliases[raw]
}

def effectiveClusteringHierarchicalEnabled(params) {
    return boolOrDefault(
        params.clustering_squidpy_hierarchical_enabled,
        true,
        "clustering_squidpy_hierarchical_enabled",
    )
}

def effectiveClusteringReferenceAtlas(params) {
    if (params.clustering_squidpy_broad_reference_atlas != null &&
        params.clustering_squidpy_broad_reference_atlas.toString().trim()) {
        return params.clustering_squidpy_broad_reference_atlas
            .toString()
            .trim()
            .toLowerCase()
    }
    return normalizeSpecies(params.species) == "mouse" ? "wmb" : "whb"
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
    if (analysisMode == "atera") {
        return ["ATERA"]
    }
    throw new IllegalArgumentException("Unknown analysis mode: ${analysisMode}")
}

// Where the cells of a row come from (plan AT-6): "native" keeps the vendor
// segmentation (ATERA), "native_proseg" also adds ProSeg cells refined from
// the vendor masks (ATERA, AT-9; no Cellpose; layer proseg_vendor),
// "cellpose_only" runs Cellpose without ProSeg (reserved; no platform allows
// it), "cellpose_proseg" refines Cellpose masks with ProSeg: the
// XENIUM/MERSCOPE route and, tiled on DAPI + 18S Cellpose masks, the ATERA
// route of AT-7 (layers reseg and proseg_mask beside the vendor original_seg).
// null or blank means the platform default of the registry. The bare word
// "proseg" keeps meaning cellpose_proseg.
def normalizeSegmentationSource(rawValue) {
    if (rawValue == null || !rawValue.toString().trim()) {
        return null
    }
    def key = rawValue
        .toString()
        .trim()
        .toLowerCase()
        .replaceAll(/[^a-z0-9]+/, "_")
        .replaceAll(/^_+|_+$/, "")
    def aliases = [
        "native": "native",
        "vendor": "native",
        "instrument": "native",
        "native_proseg": "native_proseg",
        "vendor_proseg": "native_proseg",
        "instrument_proseg": "native_proseg",
        "cellpose": "cellpose_only",
        "cellpose_only": "cellpose_only",
        "proseg": "cellpose_proseg",
        "cellpose_proseg": "cellpose_proseg",
        "cellpose_comparison": "cellpose_comparison",
    ]
    if (!aliases.containsKey(key)) {
        throw new IllegalArgumentException(
            "Unknown segmentation_source '${rawValue}'. Valid values: " +
            "native, native_proseg, cellpose_only, cellpose_proseg"
        )
    }
    return aliases[key]
}

// The segmentation source of a row: its samplesheet value, else
// --segmentation_source, else the registry default of its platforms. It must
// be one the registry allows for every active platform.
def segmentationSourceForRow(row, activePlatforms, params) {
    def requested = normalizeSegmentationSource(
        rowFieldOrDefault(row, "segmentation_source", params.segmentation_source)
    )
    def specs = activePlatforms.collect { platform -> PlatformSpec.forName(platform) }
    def source = requested
    if (source == null) {
        def defaults = specs.collect { spec -> spec.defaultSegmentationSource }.unique()
        if (defaults.size() != 1) {
            throw new IllegalArgumentException(
                "Samplesheet row ${row.pair_id} needs a segmentation_source: its " +
                "platforms ${activePlatforms.join(', ')} have different defaults " +
                "(${defaults.join(', ')})"
            )
        }
        source = defaults[0]
    }
    specs.each { spec ->
        if (!spec.segmentationSources.contains(source)) {
            throw new IllegalArgumentException(
                "segmentation_source=${source} is not available for ${spec.name} " +
                "rows yet; ${spec.name} supports: ${spec.segmentationSources.join(', ')}"
            )
        }
    }
    return source
}

// Analysis layers a segmentation source produces
// (merxen.platforms.segmentation_layers). With platforms, a platform without
// the hybrid layer (ATERA) drops proseg_hybrid; null gives the
// platform-independent list.
def availableSegmentationLayers(segmentationSource, platforms = null) {
    def layers = [
        "native": ["original_seg"],
        "native_proseg": ["proseg_vendor", "original_seg"],
        "cellpose_proseg": ["reseg", "original_seg", "proseg_mask", "proseg_hybrid"],
        "cellpose_only": ["proseg_mask", "original_seg"],
        "cellpose_comparison": ["reseg", "original_seg", "proseg_mask", "stock_cellpose"],
    ]
    if (!layers.containsKey(segmentationSource)) {
        throw new IllegalArgumentException(
            "Unknown segmentation source: ${segmentationSource}"
        )
    }
    def available = layers[segmentationSource]
    def withoutHybrid = (platforms ?: []).any { platform ->
        !PlatformSpec.forName(platform).prosegHybrid
    }
    if (withoutHybrid) {
        available = available.findAll { layer -> layer != "proseg_hybrid" }
    }
    return available
}

// Sources whose original layer is the vendor segmentation: their segment stage
// is NATIVE_LATEST and no Cellpose nuclei stage runs
// (merxen.platforms.uses_vendor_segmentation).
def usesVendorSegmentation(segmentationSource) {
    return segmentationSource in ["native", "native_proseg"]
}

// Whether a row builds its latest store with NATIVE_LATEST: vendor sources,
// and every ATERA route (registry nativeLatestBase), so the vendor
// original_seg layer of an ATERA cellpose_proseg row comes from it too
// (merxen.platforms.uses_native_latest).
def usesNativeLatest(segmentationSource, platforms) {
    return usesVendorSegmentation(segmentationSource) || (platforms ?: []).any { platform ->
        PlatformSpec.forName(platform).nativeLatestBase
    }
}

// Whether a row runs the tiled ATERA ProSeg route (plan, tiles, stitch): the
// vendor prior (native_proseg, AT-9) or ATERA Cellpose masks (cellpose_proseg,
// AT-7).
def runsAteraProseg(segmentationSource, platforms) {
    if (segmentationSource == "native_proseg") {
        return true
    }
    return segmentationSource in ["cellpose_proseg", "cellpose_comparison"] && (platforms ?: []).contains("ATERA")
}

// Whether a row runs Cellpose on the ATERA morphology image (DAPI + 18S, AT-7).
def runsAteraCellpose(segmentationSource, platforms) {
    return segmentationSource in ["cellpose_proseg", "cellpose_comparison"] && (platforms ?: []).contains("ATERA")
}

// Keep the layers a source has (AT-6 rule): a default or group value ("all",
// a global param) falls back to the layers the source has, adding a note on
// what was dropped (the workflow logs the notes of each row); a samplesheet
// value naming a layer the source lacks is an error. A default that keeps no
// layer at all (the MENDER default proseg_hybrid) resolves to every layer the
// source has (AT-9; native has one). The layers are those of the row's
// platforms (ATERA has no proseg_hybrid, AT-7). A group value drops
// proseg_vendor, the layer of the native_proseg route only, without a note
// (XENIUM and MERSCOPE rows keep their notes as they were). `strict` is a
// boolean (every missing layer is an error) or the list of layers the row
// named explicitly: only those are errors, so a group word mixed with names
// ("all,original_seg") still falls back for the members the source lacks.
// Kept layers stay in the order asked. null returns `selected` as is.
def filterSegmentationsForSource(
    selected,
    segmentationSource,
    strict,
    label,
    pairId,
    notes,
    platforms = null
) {
    if (segmentationSource == null) {
        return selected
    }
    def available = availableSegmentationLayers(segmentationSource, platforms)
    def missing = selected.findAll { segmentation -> !available.contains(segmentation) }
    if (!missing) {
        return selected
    }
    def missingNamed = strict instanceof Collection
        ? missing.findAll { segmentation -> strict.contains(segmentation) }
        : (strict ? missing : [])
    if (missingNamed) {
        throw new IllegalArgumentException(
            "${label}=${missingNamed.join(',')} for ${pairId} is not available with " +
            "segmentation_source=${segmentationSource}; available: " +
            "${available.join(', ')}"
        )
    }
    def kept = selected.findAll { segmentation -> available.contains(segmentation) }
    if (!kept) {
        kept = available
    }
    def asked = selected.findAll { segmentation -> segmentation != "proseg_vendor" }
    if (asked != kept) {
        notes << (
            "${label} ${asked.join(', ')} resolves to ${kept.join(', ')} " +
            "(segmentation_source=${segmentationSource} has ${available.join(', ')})"
        )
    }
    return kept
}

def normalizeAnalysisSegmentation(rawValue, segmentationSource = null) {
    def raw = rawValue == null ? "all" : rawValue.toString().trim()
    if (!raw) {
        raw = "all"
    }
    def aliases = [
        "both": ["reseg", "proseg_vendor", "original_seg"],
        "all": ["reseg", "proseg_vendor", "original_seg", "proseg_mask", "proseg_hybrid"],
        "reseg": ["reseg"],
        "resegmented": ["reseg"],
        "proseg": ["reseg"],
        "mosaik": ["reseg"],
        "proseg_vendor": ["proseg_vendor"],
        "vendor_proseg": ["proseg_vendor"],
        "native_proseg": ["proseg_vendor"],
        "proseg_mask": ["proseg_mask"],
        "stock_cellpose": ["stock_cellpose"],
        "cellpose": ["proseg_mask"],
        "cellpose_mask": ["proseg_mask"],
        "mask": ["proseg_mask"],
        "hybrid": ["proseg_hybrid"],
        "proseg_hybrid": ["proseg_hybrid"],
        "hybrid_seg": ["proseg_hybrid"],
        "original": ["original_seg"],
        "original_seg": ["original_seg"],
        "original_segmentation": ["original_seg"],
        "instrument": ["original_seg"],
        "instrument_seg": ["original_seg"],
        "instrument_segmentation": ["original_seg"],
    ]
    if (segmentationSource == "cellpose_comparison") aliases.all << "stock_cellpose"
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
                    "both, all, reseg, original_seg, proseg_mask, " +
                    "cellpose, proseg_hybrid, proseg_vendor"
                )
            }
            aliases[key].each { segmentation ->
                if (!selected.contains(segmentation)) {
                    selected << segmentation
                }
            }
        }
    return selected ? selected : [
        "reseg",
        "original_seg",
        "proseg_mask",
        "proseg_hybrid",
    ]
}

def normalizeMenderSegmentations(rawValue, segmentationSource = null) {
    def values = rawValue instanceof Collection
        ? rawValue
        : rawValue.toString().split(",")
    def aliases = [
        "all": ["reseg", "proseg_vendor", "original_seg", "proseg_mask", "proseg_hybrid"],
        "reseg": ["reseg"],
        "resegmented": ["reseg"],
        "proseg": ["reseg"],
        "proseg_vendor": ["proseg_vendor"],
        "vendor_proseg": ["proseg_vendor"],
        "original": ["original_seg"],
        "original_seg": ["original_seg"],
        "instrument": ["original_seg"],
        "proseg_mask": ["proseg_mask"],
        "stock_cellpose": ["stock_cellpose"],
        "cellpose": ["proseg_mask"],
        "mask": ["proseg_mask"],
        "proseg_hybrid": ["proseg_hybrid"],
        "hybrid": ["proseg_hybrid"],
    ]
    if (segmentationSource == "cellpose_comparison") aliases.all << "stock_cellpose"
    def selected = []
    values.each { value ->
        def key = value
            .toString()
            .trim()
            .toLowerCase()
            .replaceAll(/[^a-z0-9]+/, "_")
            .replaceAll(/^_+|_+$/, "")
        if (!key) {
            return
        }
        if (!aliases.containsKey(key)) {
            throw new IllegalArgumentException(
                "Unknown mender segmentation '${value}'. Valid values: " +
                "all, reseg, original_seg, proseg_mask, proseg_hybrid, proseg_vendor"
            )
        }
        aliases[key].each { segmentation ->
            if (!selected.contains(segmentation)) {
                selected << segmentation
            }
        }
    }
    if (!selected) {
        throw new IllegalArgumentException("mender_segmentations must not be empty")
    }
    return selected
}

// The layers a samplesheet value names explicitly: every token but the group
// words "all" and "both", normalised by `normalizer` (null: none named).
def explicitlyNamedSegmentations(rawValue, normalizer) {
    if (rawValue == null) {
        return []
    }
    def named = []
    rawValue.toString().split(",").each { token ->
        def key = token.trim().toLowerCase()
        if (!key || key in ["all", "both"]) {
            return
        }
        normalizer.call(key).each { segmentation ->
            if (!named.contains(segmentation)) {
                named << segmentation
            }
        }
    }
    return named
}

// The layers of a row setting (analysis_segmentation, mender_segmentations):
// `value` (the row's value, else the global param) normalised and filtered for
// the row's source; only the layers the samplesheet value names are strict.
def resolveRowSegmentations(
    rowValue,
    value,
    normalizer,
    segmentationSource,
    label,
    pairId,
    notes,
    platforms
) {
    return filterSegmentationsForSource(
        normalizer.call(value),
        segmentationSource,
        explicitlyNamedSegmentations(rowValue, normalizer),
        label,
        pairId,
        notes,
        platforms,
    )
}

def normalizeGastonSegmentations(rawValue) {
    def raw = rawValue == null ? "proseg_hybrid" : rawValue.toString().trim()
    if (!raw) {
        raw = "proseg_hybrid"
    }
    def aliases = [
        "all": ["reseg", "original_seg", "proseg_mask", "proseg_hybrid"],
        "reseg": ["reseg"],
        "resegmented": ["reseg"],
        "proseg": ["reseg"],
        "mosaik": ["reseg"],
        "original": ["original_seg"],
        "original_seg": ["original_seg"],
        "original_segmentation": ["original_seg"],
        "instrument": ["original_seg"],
        "proseg_mask": ["proseg_mask"],
        "stock_cellpose": ["stock_cellpose"],
        "cellpose": ["proseg_mask"],
        "cellpose_mask": ["proseg_mask"],
        "mask": ["proseg_mask"],
        "hybrid": ["proseg_hybrid"],
        "proseg_hybrid": ["proseg_hybrid"],
        "hybrid_seg": ["proseg_hybrid"],
    ]
    def selected = []
    raw.split(",")
        .collect { value -> value.trim() }
        .findAll { value -> value.length() > 0 }
        .each { value ->
            def key = value
                .toLowerCase()
                .replaceAll(/[^a-z0-9]+/, "_")
                .replaceAll(/^_+|_+$/, "")
            if (!aliases.containsKey(key)) {
                throw new IllegalArgumentException(
                    "Unknown gaston_segmentations '${value}'. Valid values: " +
                    "all, reseg, original_seg, proseg_mask, proseg_hybrid"
                )
            }
            aliases[key].each { segmentation ->
                if (!selected.contains(segmentation)) {
                    selected << segmentation
                }
            }
        }
    return selected ?: ["proseg_hybrid"]
}

def positiveIntegerListOrDefault(rawValue, defaultValue, label) {
    def values = rawValue instanceof Collection
        ? rawValue
        : rawValue.toString().replaceAll(/^\[|\]$/, "").split(",")
    def parsed = values
        .collect { value -> value.toString().trim() }
        .findAll { value -> value.length() > 0 }
        .collect { value -> value as int }
    if (!parsed) {
        parsed = defaultValue.collect { value -> value as int }
    }
    if (parsed.any { value -> value <= 0 }) {
        throw new IllegalArgumentException("${label} values must be positive")
    }
    return parsed
}

def nullableIntOrDefault(rawValue, defaultValue) {
    if (rawValue == null || rawValue.toString().trim().isEmpty()) {
        return defaultValue == null ? null : defaultValue as int
    }
    def raw = rawValue.toString().trim()
    return raw.toLowerCase() == "null" ? null : raw as int
}

def requirePlatformInput(row, pairId, platform) {
    def spec = PlatformSpec.forName(platform)
    def inputDir = chooseField(row, spec.inputDirFields)
    def spatialdataPath = chooseField(row, spec.spatialdataPathFields)
    if (!inputDir && !spatialdataPath) {
        error(
            "Samplesheet row for ${pairId} must provide " +
            "${spec.inputDirFields[0]} or ${spec.spatialdataPathFields[0]}"
        )
    }
    return [inputDir: inputDir, spatialdataPath: spatialdataPath]
}

def buildConfigForPlatform(row, pairId, platform) {
    def input = requirePlatformInput(row, pairId, platform)
    if (platform == "MERSCOPE") {
        def zRange = parseRange(row.merscope_z_range, 0, 6)
        def merscopeOptions = [
            z_layers: (zRange[0]..zRange[1]).collect { layer -> layer as int },
        ]
        // Only set when requested, so default build configs (and their task
        // hashes) are unchanged.
        if (boolOrDefault(
            params.merscope_allow_transform_override_mismatch,
            false,
            "merscope_allow_transform_override_mismatch"
        )) {
            merscopeOptions.allow_transform_override_mismatch = true
        }
        // Only set when requested, so default build configs (and their task
        // hashes) are unchanged.
        if (merscopeBeadRemovalRequested(row, pairId, params)) {
            def beadRemoval = [enabled: true]
            // The fill settings are written as they resolve (the fill is on in
            // hybrid mode by default), so the build JSON does not depend on the
            // defaults of the Python config. They sit inside the bead block, so
            // builds without bead removal keep their JSON and task hashes.
            def beadFill = merscopeBeadRemovalFillRequested(params)
            beadRemoval.fill = beadFill
            if (beadFill) {
                // The mode is only read with the fill, and the config refuses it
                // without.
                beadRemoval.fill_mode = merscopeBeadRemovalFillMode(params)
            }
            // Only set off the default, so CPU bead build configs (and their
            // task hashes) keep their form. The flat fill has no GPU
            // implementation and forces the CPU, so with it the device is not
            // written (it would change the hash for nothing).
            def beadDevice = merscopeBeadRemovalDevice(params)
            def beadForcesCpu = (
                beadFill && merscopeBeadRemovalFillMode(params) == "flat"
            )
            if (beadDevice != "cpu" && !beadForcesCpu) {
                beadRemoval.device = beadDevice
            }
            merscopeOptions.bead_removal = beadRemoval
        }
        return [
            dataset_name: "${pairId}_MERSCOPE",
            platform: "MERSCOPE",
            input_path: input.inputDir ?: input.spatialdataPath,
            output_path: "spatialdata_out/source_spatialdata.zarr",
            persistent_output_path: input.spatialdataPath ?: null,
            merscope_transform_path: chooseField(row, ["merscope_transform_path"]) ?: null,
            merscope: merscopeOptions,
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

    // ATERA writes the only build JSON with atera_* keys, so the XENIUM and
    // MERSCOPE configs (and their task hashes) are unchanged.
    if (platform == "ATERA") {
        return [
            dataset_name: "${pairId}_ATERA",
            platform: "ATERA",
            input_path: input.inputDir ?: input.spatialdataPath,
            output_path: "spatialdata_out/source_spatialdata.zarr",
            persistent_output_path: input.spatialdataPath ?: null,
            atera_spec_path: chooseField(row, ["atera_spec_path"]) ?: null,
            merscope: [:],
            xenium: [:],
            atera: ateraBuildOptions(params),
        ]
    }

    throw new IllegalArgumentException("Unknown platform: ${platform}")
}

def ateraBuildOptions(params) {
    def options = [
        min_qv: params.atera_min_qv == null ? null : (params.atera_min_qv as double),
        species: normalizeSpecies(params.species),
    ]
    // The watchdog cap of the build (the local executor does not cap memory).
    if (params.atera_build_max_rss_gb != null) {
        options.max_rss_gb = params.atera_build_max_rss_gb as double
    }
    return options
}

def segmentMetaForPlatform(row, platform, params) {
    def modelSettings = CellposeSettings.settings(row, platform, params)
    def extra = modelSettings ? [cellpose_settings: modelSettings] : [:]
    if (platform == "MERSCOPE") {
        def merscopeMeta = extra + [
            channels: parseChannels(
                row.merscope_channels,
                PlatformSpec.forName(platform).defaultChannels
            ),
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
        // Only for rows with bead removal on, so the segmentation config (and its
        // task hash) of every other row is unchanged.
        if (merscopeBeadRemovalRequested(row, row.pair_id?.toString()?.trim(), params)) {
            merscopeMeta.bead_mask_veto = [enabled: merscopeBeadMaskVetoRequested(params)]
        }
        return merscopeMeta
    }

    if (platform == "XENIUM") {
        return extra + [
            channels: parseChannels(
                row.xenium_channels,
                PlatformSpec.forName(platform).defaultChannels
            ),
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

    // ATERA cellpose_proseg rows (AT-7): Cellpose on the native morphology image
    // of the built store, DAPI + 18S as on XENIUM. No transform or spec path:
    // the pixel size comes from the store's spec sidecars.
    if (platform == "ATERA") {
        def spec = PlatformSpec.forName(platform)
        return extra + [
            channels: AteraDensitySettings.channels(row, params),
            image_key: spec.nativeImageKey,
            voxel_layers: intOrDefault(
                row.atera_voxel_layers,
                params.default_atera_voxel_layers
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
    def spec = PlatformSpec.forName(platform)
    // The vendor-prior ProSeg layer of native_proseg rows (AT-9, renamed in AT-7).
    if (segmentation in ["proseg_vendor", "vendor_proseg"]) {
        def layer = AteraProsegLayers.forKind("vendor")
        return [
            table_key: layer.table_key,
            shape_key: layer.shapes_key,
        ]
    }
    if (segmentation in ["proseg", "reseg"]) {
        return [
            table_key: "table_MOSAIK_proseg",
            shape_key: "MOSAIK_proseg",
        ]
    }
    if (segmentation in ["original", "original_seg"]) {
        return [
            table_key: "table_original",
            shape_key: spec.originalCellShapeKey,
        ]
    }
    if (segmentation == "proseg_hybrid") {
        return [
            table_key: "table_MOSAIK_proseg_hybrid",
            shape_key: "MOSAIK_proseg_hybrid",
        ]
    }
    if (segmentation == "stock_cellpose") {
        return [
            table_key: "table_MOSAIK_stock_cellpose",
            shape_key: "MOSAIK_stock_cellpose",
        ]
    }
    if (segmentation in ["cellpose", "proseg_mask"]) {
        return [
            table_key: "table_MOSAIK_cellpose",
            shape_key: "MOSAIK_cellpose",
        ]
    }
    if (segmentation == "proseg_geometry_assignment") {
        return [
            table_key: "table_MOSAIK_proseg_geometry_assignment",
            shape_key: "MOSAIK_proseg",
        ]
    }
    throw new IllegalArgumentException("Unknown analysis segmentation: ${segmentation}")
}

// rca-hook:H2: a map_first suffix (plan §4.8) is appended on every path; a null or
// blank suffix (legacy rows) keeps the unsuffixed key.
def clusteredSpatialdataTableKey(sourceTableKey, segmentation, suffix = "") {
    def suffixToken = suffix == null ? "" : suffix.toString().trim()
    def suffixFragment = suffixToken ? "_${suffixToken}" : ""
    if (segmentation == "reseg" || sourceTableKey == "table_MOSAIK_proseg") {
        return "table_MOSAIK_proseg_clustering_squidpy${suffixFragment}"
    }
    if (segmentation == "original_seg" || sourceTableKey == "table_original") {
        return "table_original_clustering_squidpy${suffixFragment}"
    }
    return "${sourceTableKey}_clustering_squidpy${suffixFragment}"
}

def gastonConfigJson(
    pairId,
    segmentation,
    platform,
    latestZarr,
    settings
) {
    def layerKeys = analysisLayerKeys(platform, segmentation)
    def gaston = settings.gaston
    def config = [
        pair_id: pairId,
        sample_id: "${pairId}_${platform}",
        platform: platform,
        segmentation: segmentation,
        clustered_h5ad_path: "clustered_input.h5ad",
        latest_zarr_path: latestZarr.toString(),
        source_table_key: layerKeys.table_key,
        clustered_table_key: clusteredSpatialdataTableKey(
            layerKeys.table_key,
            segmentation,
            settings.clustering_squidpy_table_key_suffix, // rca-site:H2
        ),
        shape_key: layerKeys.shape_key,
        output_dir: "gaston_out",
        use_gpu: gaston.use_gpu,
        n_restarts: gaston.n_restarts,
        epochs: gaston.epochs,
        checkpoint_interval: gaston.checkpoint_interval,
        hidden_spatial: gaston.hidden_spatial,
        hidden_expression: gaston.hidden_expression,
        optimizer: gaston.optimizer,
        glmpca_dimensions: gaston.glmpca_dimensions,
        glmpca_penalty: gaston.glmpca_penalty,
        glmpca_iterations: gaston.glmpca_iterations,
        domain_mode: gaston.domain_mode,
        num_domains: gaston.num_domains,
        min_domains: gaston.min_domains,
        max_domains: gaston.max_domains,
        domain_buckets: gaston.domain_buckets,
        auto_k_fallback: gaston.auto_k_fallback,
        write_spatialdata_table: gaston.write_spatialdata_table,
        keep_seed_models: gaston.keep_seed_models,
        keep_checkpoints: gaston.keep_checkpoints,
        max_genes: gaston.max_genes,
        figure_dpi: gaston.figure_dpi,
    ]
    return groovy.json.JsonOutput.prettyPrint(
        groovy.json.JsonOutput.toJson(config)
    )
}

def gastonTrainingConfigJson(
    pairId,
    segmentation,
    platform,
    latestZarr,
    settings
) {
    // Domain selection is postprocessing-only. Keep these values stable in the
    // configuration hashed by PREPARE, GLM-PCA, and TRAIN so changing K can
    // reuse every completed restart.
    def trainingSettings = new LinkedHashMap(settings)
    def trainingGaston = new LinkedHashMap(settings.gaston)
    trainingGaston.domain_mode = "auto"
    trainingGaston.num_domains = null
    trainingSettings.gaston = trainingGaston
    return gastonConfigJson(
        pairId,
        segmentation,
        platform,
        latestZarr,
        trainingSettings,
    )
}

def normalizeDistanceFromObjectSegmentations(rawValue) {
    def rawValues = rawValue instanceof Collection
        ? rawValue
        : rawValue.toString().split(",")
    def aliases = [
        "reseg": "proseg",
        "resegmented": "proseg",
        "proseg": "proseg",
        "original": "original",
        "original_seg": "original",
        "instrument": "original",
        "proseg_mask": "cellpose",
        "stock_cellpose": "stock_cellpose",
        "mask": "cellpose",
        "cellpose": "cellpose",
        "cellpose_mask": "cellpose",
        "proseg_geometry": "proseg_geometry_assignment",
        "proseg_geometry_assignment": "proseg_geometry_assignment",
        "hybrid": "proseg_hybrid",
        "proseg_hybrid": "proseg_hybrid",
    ]
    def selected = []
    rawValues.each { value ->
        def key = value
            .toString()
            .trim()
            .toLowerCase()
            .replaceAll(/[^a-z0-9]+/, "_")
            .replaceAll(/^_+|_+$/, "")
        if (!key) {
            return
        }
        if (!aliases.containsKey(key)) {
            throw new IllegalArgumentException(
                "Unknown distance_from_object segmentation '${value}'. Valid " +
                "values: proseg, original, cellpose, proseg_geometry_assignment"
            )
        }
        if (!selected.contains(aliases[key])) {
            selected << aliases[key]
        }
    }
    return selected ?: ["proseg", "original", "cellpose"]
}

def normalizeOptionalStringList(rawValue) {
    if (rawValue == null) {
        return null
    }
    def values = rawValue instanceof Collection
        ? rawValue
        : rawValue.toString().split(",")
    def cleaned = values
        .collect { value -> value.toString().trim() }
        .findAll { value -> value.length() > 0 }
        .unique()
    return cleaned ?: null
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
        // Whole-transcriptome ATERA panels: no 300-gene panel check, and the
        // PCA runs on highly variable genes. XENIUM/MERSCOPE samples keep
        // their JSON (and task hashes).
        if (platform == "ATERA") {
            sample.expected_panel_gene_count = null
            sample.hvg_n_top_genes = params.atera_clustering_hvg_n_top_genes == null
                ? null
                : params.atera_clustering_hvg_n_top_genes as int
        }
        if (platformPaths.containsKey(platform)) {
            sample.zarr_path = platformPaths[platform].toString()
        }
        return sample
    }
    return groovy.json.JsonOutput.prettyPrint(groovy.json.JsonOutput.toJson(samples))
}

def mergeMecrSamplesJson(samplesJsonValues) {
    def samplesByInput = [:]
    samplesJsonValues.each { samplesJson ->
        def samples = new groovy.json.JsonSlurper().parseText(samplesJson.toString())
        samples.each { sample ->
            def inputKey = [sample.zarr_path, sample.table_key].join("|")
            samplesByInput[inputKey] = sample
        }
    }
    def mergedSamples = samplesByInput.values().toList().sort { sample ->
        [sample.sample_id, sample.table_key, sample.zarr_path].join("|")
    }
    return groovy.json.JsonOutput.prettyPrint(
        groovy.json.JsonOutput.toJson(mergedSamples)
    )
}

// Whether a MECR sample list belongs to an ATERA row, whose reference streams
// (MECR_REFERENCE_ATERA). ATERA rows analyse one platform, so no list mixes
// ATERA with XENIUM or MERSCOPE samples.
def mecrSamplesUseAteraReference(samplesJson) {
    def samples = new groovy.json.JsonSlurper().parseText(samplesJson.toString())
    return samples.any { sample -> sample.platform?.toString() == "ATERA" }
}

// One reference task over every MECR sample list of a channel (none when it is
// empty): the atlas for the run's species and the merged sample JSON.
def mecrReferenceInputs(samplesChannel) {
    return samplesChannel
        .map { _pairId, _segmentation, samplesJson -> samplesJson }
        .collect()
        .filter { samplesJsonValues -> !samplesJsonValues.isEmpty() }
        .map { samplesJsonValues ->
            def referenceAtlas = normalizeSpecies(params.species) == "mouse"
                ? "wmb"
                : "whb"
            tuple(referenceAtlas, mergeMecrSamplesJson(samplesJsonValues))
        }
}

def spatialGeneSamplesJson(samplesJson, row) {
    def samples = new groovy.json.JsonSlurper().parseText(samplesJson.toString())
    samples.each { sample ->
        def platform = sample.platform.toString()
        def sampleSpec = PlatformSpec.forName(platform)
        // Native rows have no Cellpose nuclei; their vendor nuclei are the
        // registry's original nucleus shapes.
        sample.nuclei_shape_key = sampleSpec.defaultSegmentationSource == "native"
            ? sampleSpec.originalNucleusShapeKey
            : "cellpose_nuclei"
        sample.pial_boundary_path = optionalNormalizedPathString(
            corticalDepthAnnotationPath(row, platform, "pial")
        )
        sample.wm_boundary_path = optionalNormalizedPathString(
            corticalDepthAnnotationPath(row, platform, "wm")
        )
        sample.side_boundary_path = optionalNormalizedPathString(
            corticalDepthAnnotationPath(row, platform, "side")
        )
        sample.exclusion_path = optionalNormalizedPathString(
            corticalDepthAnnotationPath(row, platform, "exclusion")
        )
        sample.ribbon_path = optionalNormalizedPathString(
            corticalDepthAnnotationPath(row, platform, "ribbon")
        )
        sample.annotation_path = optionalNormalizedPathString(
            corticalDepthAnnotationPath(row, platform, "annotation")
        )
    }
    return groovy.json.JsonOutput.prettyPrint(
        groovy.json.JsonOutput.toJson(samples)
    )
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
        ] + AnnotationSettings.clusteredTableFields(row, params) // rca-site:H2
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
        boundary_frame: corticalDepthBoundaryFrame(row, platform, params),
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

def distanceFromObjectConfigForPlatform(
    pairId,
    platform,
    segmentations,
    params
) {
    def tables = segmentations.collect { segmentation ->
        def layerKeys = analysisLayerKeys(platform, segmentation)
        [
            segmentation: segmentation,
            table_key: layerKeys.table_key,
            shape_key: layerKeys.shape_key,
        ]
    }
    return [
        pair_id: pairId,
        dataset_name: "${pairId}_${platform}",
        platform: platform,
        latest_zarr_path: "latest_input.zarr",
        output_dir: "distance_from_object_out",
        object_annotation_path: "object_annotations.geojson",
        tables: tables,
        object_types: normalizeOptionalStringList(
            params.distance_from_object_object_types
        ),
        coordinate_unit_um: params.distance_from_object_coordinate_unit_um,
        near_distance_um: params.distance_from_object_near_distance_um,
        far_distance_um: params.distance_from_object_far_distance_um,
        max_distance_um: params.distance_from_object_max_distance_um,
        tissue_annotation_column: "cortical_depth_annotation",
        included_tissue_annotations: ["grey_matter"],
        min_cells_per_pseudobulk: params.distance_from_object_min_cells_per_pseudobulk,
        write_spatialdata_table: params.distance_from_object_write_spatialdata_table,
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
        "train_cellpose": "train_cellpose",
        "cellpose_hitl_train": "train_cellpose",
        "segment_nuclei": "segment_nuclei",
        "nuclei_segment": "segment_nuclei",
        "cellpose_nuclei": "segment_nuclei",
        "nuclei_cellpose": "segment_nuclei",
        "enrich": "enrich",
        "enrichment": "enrich",
        "register_images": "register_images",
        "register_image": "register_images",
        "image_registration": "register_images",
        "registered_images": "register_images",
        "build_viewer_caches": "build_viewer_caches",
        "viewer_caches": "build_viewer_caches",
        "viewer_cache": "build_viewer_caches",
        "pyramids": "build_viewer_caches",
        "mask_image_quantification": "mask_image_quantification",
        "image_quantification": "mask_image_quantification",
        "quantify_images": "mask_image_quantification",
        "mask_quantification": "mask_image_quantification",
        "compute_cortical_depth": "compute_cortical_depth",
        "cortical_depth": "compute_cortical_depth",
        "laplace_depth": "compute_cortical_depth",
        "distance_from_object": "distance_from_object",
        "object_distance": "distance_from_object",
        "distance_to_object": "distance_from_object",
        "plaque_distance": "distance_from_object",
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
        "spatial_gene_analysis": "spatial_gene_analysis",
        "spatial_genes": "spatial_gene_analysis",
        "gene_spatial_analysis": "spatial_gene_analysis",
        "spatial_autocorrelation": "spatial_gene_analysis",
        "mecr": "mecr",
        "mutually_exclusive_coexpression": "mecr",
        "mutually_exclusive_co_expression": "mecr",
        "cluster": "clustering_squidpy",
        "clustering": "clustering_squidpy",
        "clustering_squidpy": "clustering_squidpy",
        "squidpy": "clustering_squidpy",
        "pciseq": "pciseq",
        "pci_seq": "pciseq",
        "cell_type_mapping": "mapmycells",
        "celltype_mapping": "mapmycells",
        "map_my_cells": "mapmycells",
        "mapmycells": "mapmycells",
        "mender": "mender",
        "spatial_domains": "mender",
        "gaston": "gaston",
        "topography": "gaston",
        "topographic_domains": "gaston",
    ]
    if (!aliases.containsKey(key)) {
        throw new IllegalArgumentException(
            "Unknown ${paramName} '${raw}'. Valid stages: " +
            "build_spatialdata, segment_nuclei, segment, enrich, register_images, " +
            "build_viewer_caches, mask_image_quantification, " +
            "compute_cortical_depth, distance_from_object, qc, align, align_qc, " +
            "compare, visualize, spatial_gene_analysis, mecr, " +
            "clustering_squidpy, pciseq, mender, mapmycells, gaston"
        )
    }
    return aliases[key]
}

def activeStageOrder(
    alignmentEnabled,
    pairedMode,
    maskImageQuantificationEnabled,
    spatialGeneAnalysisEnabled,
    corticalDepthEnabled,
    distanceFromObjectEnabled,
    viewerCacheEnabled,
    mecrEnabled,
    menderEnabled,
    registerImagesEnabled,
    gastonEnabled = false,
    pciseqEnabled = false,
    segmentationSource = "cellpose_proseg",
    platforms = null,
    cellposeHitlTrainingEnabled = false
) {
    def stages = ["build_spatialdata", "segment_nuclei", "segment", "enrich"]
    // Native rows and every ATERA row keep the vendor nuclei: "segment" is
    // NATIVE_LATEST (and, for native_proseg and ATERA cellpose_proseg, the
    // ATERA Cellpose and tiled ProSeg tasks) and no Cellpose nuclei stage runs.
    if (usesNativeLatest(segmentationSource, platforms)) {
        stages = ["build_spatialdata", "segment", "enrich"]
    }
    if (cellposeHitlTrainingEnabled) {
        stages.add(stages.indexOf('segment'), 'train_cellpose')
    }
    // Registered images join the zarr before the viewer cache and quantification
    // so both see the new channels.
    if (registerImagesEnabled) {
        stages += ["register_images"]
    }
    if (viewerCacheEnabled) {
        stages += ["build_viewer_caches"]
    }
    if (maskImageQuantificationEnabled) {
        stages += ["mask_image_quantification"]
    }
    stages += ["qc"]
    if (mecrEnabled) {
        stages += ["mecr"]
    }
    if (pairedMode && alignmentEnabled) {
        stages += ["align", "align_qc"]
    }
    if (pairedMode) {
        stages += ["compare"]
    }
    stages += ["visualize"]
    if (spatialGeneAnalysisEnabled) {
        stages += ["spatial_gene_analysis"]
    }
    stages += ["clustering_squidpy"]
    if (pciseqEnabled) {
        stages += ["pciseq"]
    }
    // compute_cortical_depth is sequenced AFTER clustering_squidpy so the per-cell
    // cluster annotations exist for its depth violin plots. It is a terminal
    // opt-in stage; see the default stop_stage handling in rowSampleSettings.
    if (corticalDepthEnabled) {
        stages += ["compute_cortical_depth"]
    }
    if (distanceFromObjectEnabled) {
        stages += ["distance_from_object"]
    }
    stages += ["mapmycells"]
    if (menderEnabled) {
        stages += ["mender"]
    }
    if (gastonEnabled) {
        stages += ["gaston"]
    }
    return stages
}

def validateStage(
    stage,
    stages,
    paramName,
    alignmentEnabled,
    spatialGeneAnalysisEnabled,
    menderEnabled,
    segmentationSource = "cellpose_proseg",
    ateraRow = false
) {
    if (!stages.contains(stage)) {
        def hint = ""
        if (stage == "segment_nuclei" && usesVendorSegmentation(segmentationSource)) {
            hint = (
                " Rows with segmentation_source=${segmentationSource} keep the vendor " +
                "nuclei and run no Cellpose nuclei stage; use segment (NATIVE_LATEST)."
            )
        } else if (stage == "segment_nuclei" && ateraRow) {
            hint = (
                " ATERA rows keep the vendor nuclei and run no Cellpose nuclei stage; " +
                "use segment."
            )
        }
        if (stage in ["align", "align_qc"] && !alignmentEnabled) {
            hint = " Pass --enable_alignment true to use alignment stages."
        }
        if (stage == "compute_cortical_depth") {
            hint = " Pass --cortical_depth_enabled true to use cortical depth."
        }
        if (stage == "distance_from_object") {
            hint = " Pass --distance_from_object_enabled true to use object distance."
        }
        if (stage == "mecr") {
            hint = " Pass --mecr_enabled true to use MECR."
        }
        if (stage == "spatial_gene_analysis" && !spatialGeneAnalysisEnabled) {
            hint = " Pass --spatial_gene_analysis_enabled true to use spatial gene analysis."
        }
        if (stage == "mender" && !menderEnabled) {
            hint = " Pass --mender_enabled true to use MENDER."
        }
        if (stage == "register_images") {
            hint = " Set xenium_registered_images_csv for this Xenium row to register images."
        }
        if (stage == "gaston") {
            hint = " Pass --gaston_enabled true to use GASTON."
        }
        if (stage == "pciseq") {
            hint = " Pass --pciseq_enabled true to use pciSeq."
        }
        // The enable hints above lead into the ATERA refusal; ATERA rows are
        // told which stages they run instead (the native nuclei hint stays).
        if (ateraRow && !(stage in ateraNativeStages()) && stage != "segment_nuclei") {
            hint = " ATERA rows run only ${ateraNativeStages().join(', ')}."
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

def currentPairTerminalStage(settings) {
    if (settings.gaston_bypass_terminal_gate || settings.mender_bypass_terminal_gate) {
        return null
    }
    if (settings.run_mapmycells) {
        return "mapmycells"
    }
    if (settings.run_distance_from_object) {
        // DISTANCE_FROM_OBJECT_COHORT is intentionally outside this barrier.
        return "distance_from_object_annotation"
    }
    if (settings.run_compute_cortical_depth) {
        return "compute_cortical_depth"
    }
    if (settings.run_clustering_squidpy) {
        return "clustering_squidpy"
    }
    return null
}

def currentPairTerminalExpectedCount(settings, terminalStage) {
    if (terminalStage == "mapmycells") {
        def extraClustering = settings.required_clustering_segmentations.findAll {
            segmentation -> !settings.analysis_segmentations.contains(segmentation)
        }
        return settings.analysis_segmentations.size() + extraClustering.size()
    }
    if (terminalStage in [
        "distance_from_object_annotation",
        "compute_cortical_depth",
    ]) {
        return settings.active_platforms.size()
    }
    if (terminalStage == "clustering_squidpy") {
        return settings.required_clustering_segmentations.size()
    }
    return 0
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

def findCachedAtlasFilePath(rawCacheDir, referenceAtlas, filename) {
    if (isBlankPath(rawCacheDir)) {
        return null
    }
    def root = normalizedPath(rawCacheDir)
    if (!root.toFile().isDirectory()) {
        return null
    }
    def taxonomyDir = referenceAtlas == "wmb" ? "WMB-taxonomy" : "WHB-taxonomy"
    def matches = []
    root.toFile().eachFileRecurse { candidate ->
        if (candidate.isFile() && candidate.name == filename) {
            if (filename.endsWith(".json") || candidate.toPath().any { part ->
                part.toString() == taxonomyDir
            }) {
                matches << candidate.toPath()
            }
        }
    }
    return matches ? matches.sort { path -> path.toString() }[-1] : null
}

def appendClusteringSquidpyPreflightChecks(errors, settings, params) {
    // rca-hook:H4: map_first checks (plan §3.7); the checks below are legacy-only.
    AnnotationPreflight.append(errors, settings, params)
    if (!AnnotationSettings.isLegacy(settings)) {
        return
    }
    def hierarchicalEnabled = effectiveClusteringHierarchicalEnabled(params)
    if (!(settings.run_clustering_squidpy && hierarchicalEnabled)) {
        return
    }
    def referenceAtlas = effectiveClusteringReferenceAtlas(params)
    def expectedAtlas = settings.species == "mouse" ? "wmb" : "whb"
    if (referenceAtlas != expectedAtlas) {
        errors << (
            "CLUSTERING_SQUIDPY species ${settings.species} requires " +
            "--clustering_squidpy_broad_reference_atlas ${expectedAtlas}"
        )
        return
    }
    def markerLookupPath = params.clustering_squidpy_broad_marker_lookup_path ?:
        (referenceAtlas == "whb" ? params.clustering_squidpy_whb_marker_lookup_path : null)
    def taxonomyMetadataPath = params.clustering_squidpy_broad_taxonomy_metadata_path ?:
        (referenceAtlas == "whb" ? params.clustering_squidpy_whb_taxonomy_metadata_path : null)
    def clusterMembershipPath = params.clustering_squidpy_broad_cluster_membership_path ?:
        (referenceAtlas == "whb" ? params.clustering_squidpy_whb_cluster_membership_path : null)
    def autoDownloadReference = boolOrDefault(
        params.clustering_squidpy_broad_auto_download_reference,
        true,
        "clustering_squidpy_broad_auto_download_reference",
    )
    def referenceCacheDir = isBlankPath(
        params.clustering_squidpy_broad_reference_cache_dir
    ) ? normalizedPath(params.outdir).resolve("mapmycells_cache") :
        params.clustering_squidpy_broad_reference_cache_dir
    if (isBlankPath(markerLookupPath) &&
        !(referenceAtlas == "wmb" && autoDownloadReference)) {
        def markerFilename = referenceAtlas == "wmb"
            ? "mouse_markers_230821.json"
            : "query_markers.n10.20240221800.json"
        def cachedMarker = findCachedAtlasFilePath(
            referenceCacheDir,
            referenceAtlas,
            markerFilename,
        )
        if (cachedMarker == null) {
            errors << (
                "Missing CLUSTERING_SQUIDPY ${referenceAtlas.toUpperCase()} " +
                "broad marker lookup: set " +
                "--clustering_squidpy_broad_marker_lookup_path or populate " +
                "${referenceCacheDir} with ${markerFilename}"
            )
        }
    } else if (!isBlankPath(markerLookupPath)) {
        appendPreflightFileCheck(
            errors,
            markerLookupPath,
            "CLUSTERING_SQUIDPY broad marker lookup",
        )
    }
    if (isBlankPath(taxonomyMetadataPath) &&
        !(referenceAtlas == "wmb" && autoDownloadReference)) {
        def cachedTaxonomy = findCachedAtlasFilePath(
            referenceCacheDir,
            referenceAtlas,
            "cluster_annotation_term.csv",
        )
        if (cachedTaxonomy == null) {
            errors << (
                "Missing CLUSTERING_SQUIDPY ${referenceAtlas.toUpperCase()} " +
                "broad taxonomy metadata: " +
                "set --clustering_squidpy_broad_taxonomy_metadata_path " +
                "or provide a cache containing " +
                "${referenceAtlas == 'wmb' ? 'WMB-taxonomy' : 'WHB-taxonomy'}" +
                "/**/cluster_annotation_term.csv"
            )
        }
    } else if (!isBlankPath(taxonomyMetadataPath)) {
        appendPreflightFileCheck(
            errors,
            taxonomyMetadataPath,
            "CLUSTERING_SQUIDPY broad taxonomy metadata",
        )
    }
    if (!isBlankPath(clusterMembershipPath)) {
        appendPreflightFileCheck(
            errors,
            clusterMembershipPath,
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
        ? (settings.species == "mouse" ? "whole_brain" : "both")
        : params.mapmycells_reference_mode.toString().trim().toLowerCase()
    def referenceAtlas = params.mapmycells_reference_atlas == null
        ? (settings.species == "mouse" ? "wmb" : "whb")
        : params.mapmycells_reference_atlas.toString().trim().toLowerCase()
    def markerLookupPath = params.mapmycells_marker_lookup_path ?:
        (referenceAtlas == "whb" ? params.mapmycells_whb_marker_lookup_path : null)
    def precomputedStatsPath = params.mapmycells_precomputed_stats_path ?:
        (referenceAtlas == "whb" ? params.mapmycells_whb_precomputed_stats_path : null)
    def autoDownloadReferences = params.mapmycells_auto_download_references == null
        ? true
        : params.mapmycells_auto_download_references.toString().trim().toLowerCase() == "true"
    if (referenceMode in ["whole_brain", "both"]) {
        if (!autoDownloadReferences || markerLookupPath) {
            appendPreflightFileCheck(
                errors,
                markerLookupPath,
                "MAPMYCELLS whole-brain marker lookup",
            )
        }
        if (!autoDownloadReferences || precomputedStatsPath) {
            appendPreflightFileCheck(
                errors,
                precomputedStatsPath,
                "MAPMYCELLS whole-brain precomputed stats",
            )
        }
    }
    if (params.mapmycells_gene_mapping_db_path) {
        appendPreflightFileCheck(
            errors,
            params.mapmycells_gene_mapping_db_path,
            "MAPMYCELLS gene mapping database",
        )
    }
    def geneIdFallbackPath = params.annotation_gene_id_fallback_csv
    if (!isBlankPath(geneIdFallbackPath) &&
        geneIdFallbackPath.toString().trim().toLowerCase() != "false") {
        appendPreflightFileCheck(
            errors,
            geneIdFallbackPath,
            "MAPMYCELLS gene-ID fallback table (annotation_gene_id_fallback_csv)",
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

// Frame the platform's depth boundary GeoJSONs are drawn in: "native" (the
// section's own dataset microns, the default and how the annotations are
// drawn) or "aligned" (the pair's fixed-section frame). A row-level
// <platform>_cortical_depth_boundary_frame / cortical_depth_boundary_frame
// value overrides --cortical_depth_boundary_frame.
def corticalDepthBoundaryFrame(row, platform, params) {
    def prefix = platform.toString().toLowerCase()
    def raw = chooseField(row, [
        "${prefix}_cortical_depth_boundary_frame",
        "cortical_depth_boundary_frame",
    ])
    if (raw == null) {
        raw = params.cortical_depth_boundary_frame
    }
    def frame = raw == null ? "" : raw.toString().trim().toLowerCase()
    return frame ?: "native"
}

def alignmentTissueAnnotationPath(row, platform) {
    def prefix = platform.toString().toLowerCase()
    return chooseField(row, [
        "${prefix}_cortical_depth_annotation_geojson",
        "${prefix}_cortical_depth_annotations_geojson",
        "${prefix}_cortical_depth_annotation_path",
    ])
}

def distanceFromObjectAnnotationPath(row, platform) {
    def prefix = platform.toString().toLowerCase()
    return chooseField(row, [
        "${prefix}_distance_object_annotation_geojson",
        "${prefix}_distance_from_object_annotation_geojson",
        "${prefix}_object_annotation_geojson",
        "${prefix}_plaque_annotation_geojson",
        "distance_object_annotation_geojson",
        "distance_from_object_annotation_geojson",
        "object_annotation_geojson",
        "plaque_annotation_geojson",
    ])
}

def appendCorticalDepthPreflightChecks(errors, row, settings, _params) {
    if (!settings.run_compute_cortical_depth) {
        return
    }
    settings.active_platforms.each { platform ->
        def labelPrefix = "COMPUTE_CORTICAL_DEPTH ${settings.pair_id}:${platform}"
        def boundaryFrame = corticalDepthBoundaryFrame(row, platform, _params)
        if (!(boundaryFrame in ["native", "aligned"])) {
            errors << (
                "${labelPrefix} cortical_depth_boundary_frame must be " +
                "'native' or 'aligned', got '${boundaryFrame}'"
            )
        }
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

def appendAlignmentAnnotationPreflightChecks(errors, row, settings, params) {
    if (
        !settings.run_align ||
        params.alignment_backend.toString().trim().toLowerCase() != "valis"
    ) {
        return
    }
    ["MERSCOPE", "XENIUM"].each { platform ->
        appendPreflightFileCheck(
            errors,
            alignmentTissueAnnotationPath(row, platform),
            "ALIGN ${settings.pair_id}:${platform} combined pia/tissue-edge " +
                "annotation GeoJSON",
        )
    }
}

// Whether a platform's transcript spatial patterns run on the label-raster
// engine (Python selects it when the layer has cell and nucleus label
// elements: the vendor-label platforms, ATERA). That engine needs no tissue
// annotations: it derives the tissue from the all-transcript density.
def transcriptPatternsUseLabelRasters(platform) {
    return PlatformSpec.forName(platform).defaultSegmentationSource == "native"
}

def appendSpatialGeneTranscriptPreflightChecks(errors, row, settings, _params) {
    if (
        !settings.run_spatial_gene_analysis ||
        !settings.spatial_gene_analysis_transcript_analysis_enabled
    ) {
        return
    }
    settings.active_platforms.each { platform ->
        def labelPrefix = "SPATIAL_GENE_ANALYSIS ${settings.pair_id}:${platform}"
        if (transcriptPatternsUseLabelRasters(platform)) {
            // The label-raster engine derives the tissue from the transcript
            // density when a row has no annotations; given ones must exist.
            [
                annotation: "combined pia/tissue-edge annotation GeoJSON",
                pial: "pial boundary GeoJSON",
                side: "tissue-edge GeoJSON",
                wm: "wm annotation GeoJSON",
                exclusion: "exclusion annotation GeoJSON",
                ribbon: "ribbon annotation GeoJSON",
            ].each { role, description ->
                def optionalPath = corticalDepthAnnotationPath(row, platform, role)
                if (!isBlankPath(optionalPath)) {
                    appendPreflightFileCheck(
                        errors,
                        optionalPath,
                        "${labelPrefix} ${description}",
                    )
                }
            }
            return
        }
        def annotationPath = corticalDepthAnnotationPath(row, platform, "annotation")
        if (!isBlankPath(annotationPath)) {
            appendPreflightFileCheck(
                errors,
                annotationPath,
                "${labelPrefix} combined pia/tissue-edge annotation GeoJSON",
            )
        } else {
            appendPreflightFileCheck(
                errors,
                corticalDepthAnnotationPath(row, platform, "pial"),
                "${labelPrefix} pial boundary GeoJSON",
            )
            appendPreflightFileCheck(
                errors,
                corticalDepthAnnotationPath(row, platform, "side"),
                "${labelPrefix} tissue-edge GeoJSON",
            )
        }
        ["wm", "exclusion", "ribbon"].each { role ->
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

def appendDistanceFromObjectPreflightChecks(errors, row, settings, _params) {
    if (!settings.run_distance_from_object) {
        return
    }
    settings.active_platforms.each { platform ->
        appendPreflightFileCheck(
            errors,
            distanceFromObjectAnnotationPath(row, platform),
            "DISTANCE_FROM_OBJECT ${settings.pair_id}:${platform} object GeoJSON",
        )
    }
}

def appendMecrPreflightChecks(errors, settings, params) {
    if (!settings.run_mecr) {
        return
    }
    def referenceAtlas = settings.species == "mouse" ? "wmb" : "whb"
    def autoDownloadReference = boolOrDefault(
        params.mecr_auto_download_reference,
        true,
        "mecr_auto_download_reference",
    )
    def referenceH5adPaths = normalizeOptionalStringList(
        params.mecr_reference_h5ad_paths
    ) ?: []
    def neuronsPath = params.mecr_neurons_h5ad_path ?:
        (referenceAtlas == "whb" ? params.mecr_whb_neurons_h5ad_path : null)
    def nonneuronsPath = params.mecr_nonneurons_h5ad_path ?:
        (referenceAtlas == "whb" ? params.mecr_whb_nonneurons_h5ad_path : null)
    def cellMetadataPath = params.mecr_cell_metadata_path ?:
        (referenceAtlas == "whb" ? params.mecr_whb_cell_metadata_path : null)
    def taxonomyMetadataPath = params.mecr_taxonomy_metadata_path ?:
        (referenceAtlas == "whb" ? params.mecr_whb_taxonomy_metadata_path : null)
    def clusterMembershipPath = params.mecr_cluster_membership_path ?:
        (referenceAtlas == "whb" ? params.mecr_whb_cluster_membership_path : null)
    if (referenceAtlas == "wmb" && autoDownloadReference) {
        referenceH5adPaths.eachWithIndex { rawPath, index ->
            appendPreflightFileCheck(
                errors,
                rawPath,
                "MECR WMB raw H5AD ${index + 1}",
            )
        }
        [
            [cellMetadataPath, "WMB cell metadata"],
            [taxonomyMetadataPath, "WMB taxonomy metadata"],
            [clusterMembershipPath, "WMB cluster membership metadata"],
        ].each { rawPath, label ->
            if (!isBlankPath(rawPath)) {
                appendPreflightFileCheck(errors, rawPath, "MECR ${label}")
            }
        }
        return
    }
    referenceH5adPaths.eachWithIndex { rawPath, index ->
        appendPreflightFileCheck(
            errors,
            rawPath,
            "MECR ${referenceAtlas.toUpperCase()} raw H5AD ${index + 1}",
        )
    }
    if (!referenceH5adPaths) {
        [
            [neuronsPath, "${referenceAtlas.toUpperCase()} neuron raw H5AD"],
            [nonneuronsPath, "${referenceAtlas.toUpperCase()} non-neuron raw H5AD"],
        ].each { rawPath, label ->
            appendPreflightFileCheck(errors, rawPath, "MECR ${label}")
        }
    }
    [
        [cellMetadataPath, "${referenceAtlas.toUpperCase()} cell metadata"],
        [taxonomyMetadataPath, "${referenceAtlas.toUpperCase()} taxonomy metadata"],
        [clusterMembershipPath, "${referenceAtlas.toUpperCase()} cluster membership metadata"],
    ].each { rawPath, label ->
        appendPreflightFileCheck(errors, rawPath, "MECR ${label}")
    }
}

def appendMenderPreflightChecks(errors, settings, params) {
    if (!(settings.run_mender && settings.mender_published_output_mode)) {
        return
    }
    settings.mender_segmentations.each { segmentation ->
        settings.active_platforms.each { platform ->
            def sampleId = "${settings.pair_id}_${platform}"
            def clusteredH5ad = normalizedPath(
                publishedPairPath(
                    params.outdir,
                    settings.pair_id,
                    "${segmentation}/" +
                    AnnotationSettings.publishedClusteringDir(
                        params.outdir, settings.pair_id, segmentation, platform, sampleId,
                        settings.clustering_squidpy_table_key_suffix, // rca-site:H2
                    ) +
                    "/clustering_squidpy_out/${platform.toLowerCase()}/" +
                    "${sampleId}_clustered.h5ad",
                )
            )
            if (!clusteredH5ad.toFile().isFile()) {
                errors << (
                    "Missing MENDER clustered H5AD for " +
                    "${settings.pair_id}:${platform}:${segmentation}: ${clusteredH5ad}"
                )
            }
            def latestZarr = normalizedPath(
                publishedDatasetPath(
                    params.outdir,
                    settings.pair_id,
                    platform,
                    "latest/latest_spatialdata.zarr",
                )
            )
            if (!latestZarr.toFile().isDirectory()) {
                errors << (
                    "Missing MENDER latest SpatialData Zarr for " +
                    "${settings.pair_id}:${platform}:${segmentation}: ${latestZarr}"
                )
            } else {
                def layerKeys = analysisLayerKeys(platform, segmentation)
                def tableKey = clusteredSpatialdataTableKey(
                    layerKeys.table_key,
                    segmentation,
                    settings.clustering_squidpy_table_key_suffix, // rca-site:H2
                )
                def tablePath = latestZarr.resolve("tables").resolve(tableKey)
                if (!tablePath.toFile().exists()) {
                    errors << (
                        "Missing MENDER clustered SpatialData table '${tableKey}' " +
                        "for ${settings.pair_id}:${platform}:${segmentation}: " +
                        "${tablePath}"
                    )
                }
            }
        }
    }
}

def registeredImageSpecs(rawCsvPath) {
    // One row per image: image_key, image_path, alignment_matrix, and optional
    // channel_names (";"-separated) and registration_channel ("none" disables
    // refinement). Relative paths resolve against the CSV's own directory.
    def csvPath = normalizedPath(rawCsvPath)
    def manifestRows = csvPath.splitCsv(header: true, strip: true)
    if (!manifestRows) {
        throw new IllegalArgumentException(
            "Registered-images CSV ${csvPath} lists no images"
        )
    }
    def specs = []
    manifestRows.eachWithIndex { manifestRow, index ->
        def label = "${csvPath} row ${index + 2}"
        def imageKey = chooseField(manifestRow, ["image_key"])
        def imagePath = chooseField(manifestRow, ["image_path"])
        def matrixPath = chooseField(
            manifestRow,
            ["alignment_matrix", "alignment_matrix_path"],
        )
        if (!imageKey || !imagePath || !matrixPath) {
            throw new IllegalArgumentException(
                "${label} must set image_key, image_path and alignment_matrix"
            )
        }
        def channelNames = chooseField(manifestRow, ["channel_names"])
        def registrationChannel = chooseField(
            manifestRow,
            ["registration_channel"],
        ) ?: "DAPI"
        specs << [
            image_key: imageKey,
            image_path: csvPath.parent.resolve(imagePath).normalize().toString(),
            alignment_matrix_path: csvPath.parent.resolve(matrixPath).normalize().toString(),
            channel_names: channelNames
                ? channelNames.split(";").collect { name -> name.trim() }.findAll { name -> name }
                : null,
            registration_channel: registrationChannel.toLowerCase() == "none"
                ? null
                : registrationChannel,
        ]
    }
    return specs
}

def appendRegisterImagesPreflightChecks(errors, settings) {
    if (!settings.run_register_images) {
        return
    }
    def csvPath = settings.registered_images_csv
    if (!normalizedPath(csvPath).toFile().exists()) {
        errors << "Missing xenium_registered_images_csv: ${csvPath}"
        return
    }
    def specs = []
    try {
        specs = registeredImageSpecs(csvPath)
    } catch (IllegalArgumentException exc) {
        errors << exc.message
        return
    }
    def keys = specs.collect { spec -> spec.image_key }
    if (keys.unique(false).size() != keys.size()) {
        errors << "Duplicate image_key values in ${csvPath}: ${keys}"
    }
    specs.each { spec ->
        appendPreflightFileCheck(
            errors,
            spec.image_path,
            "registered image ${spec.image_key}",
        )
        appendPreflightFileCheck(
            errors,
            spec.alignment_matrix_path,
            "alignment matrix for ${spec.image_key}",
        )
    }
}

def appendMerscopeBeadRemovalPreflightChecks(errors, row, settings, params) {
    // Rows without a MERSCOPE input never use the bead config.
    if (!("MERSCOPE" in settings.active_platforms)) {
        return
    }
    if (!settings.run_build) {
        // The build is skipped: the published source zarr is used as it is.
        appendPublishedBeadRemovalChecks(errors, row, settings, params)
        return
    }
    if (!merscopeBeadRemovalRequested(row, settings.pair_id, params)) {
        return
    }
    def beadDevice = merscopeBeadRemovalDevice(params)
    if (!(beadDevice in ["auto", "cpu", "cuda"])) {
        errors << (
            "merscope_bead_removal_device must be auto, cpu or cuda; " +
            "got '${params.merscope_bead_removal_device}'."
        )
    }
    // The mode is read only with the fill (it is ignored when the fill is off),
    // but a misspelt value is refused in any case.
    def beadFillMode = merscopeBeadRemovalFillMode(params)
    if (!(beadFillMode in ["flat", "hybrid"])) {
        errors << (
            "merscope_bead_removal_fill_mode must be flat or hybrid; " +
            "got '${params.merscope_bead_removal_fill_mode}'."
        )
    }
    def merscopeDir = chooseField(row, ["merscope_dir"])
    // Without a raw folder nothing is projected (a cached zarr is reused), so
    // the z range and the raw folder only matter when merscope_dir is set.
    if (!merscopeDir) {
        return
    }
    def zRange = parseRange(row.merscope_z_range, 0, 6)
    if (zRange[0] < 1) {
        errors << (
            "merscope_bead_removal needs merscope_z_range to start at plane 1 " +
            "or later (plane 0 is the bead plane and is read for detection " +
            "only); got ${zRange[0]}-${zRange[1]}. Use for example 1-6."
        )
    }
    // Plane 7 is allowed here only if the dataset has it. That is checked
    // against the planes present by the calibration and the projection driver.
    if (zRange[1] > 7) {
        errors << (
            "merscope_bead_removal supports planes up to 7; " +
            "got ${zRange[0]}-${zRange[1]}."
        )
    }
    if (!normalizedPath(merscopeDir).toFile().isDirectory()) {
        errors << (
            "merscope_bead_removal needs a raw MERSCOPE input folder with " +
            "z-plane TIFFs (images/mosaic_<stain>_z<N>.tif); merscope_dir " +
            "is not a directory: ${merscopeDir}"
        )
    }
}

def onOffText(flag) {
    return flag ? "on" : "off"
}

def publishedBeadRemovalState(zarrPath) {
    // Bead removal and fill switches recorded in the root attribute
    // merxen_image_preprocessing of a built source zarr. A legacy zarr (no
    // record) or unreadable metadata reads as "off", like in the Python guard.
    def root = zarrPath.toFile()
    def attrs = null
    try {
        def v3 = new File(root, "zarr.json")
        def v2 = new File(root, ".zattrs")
        if (v3.isFile()) {
            attrs = new groovy.json.JsonSlurper().parseText(v3.text)?.attributes
        } else if (v2.isFile()) {
            attrs = new groovy.json.JsonSlurper().parseText(v2.text)
        }
    } catch (Exception ignored) {
        attrs = null
    }
    def record = (attrs instanceof Map) ? attrs.get("merxen_image_preprocessing") : null
    def bead = (record instanceof Map) ? record.get("bead_removal") : null
    def fill = (bead instanceof Map) ? bead.get("fill") : null
    return [
        enabled: (bead instanceof Map) && bead.get("enabled") == true,
        fill: (fill instanceof Map) && fill.get("enabled") == true,
        // A fill block without a mode is a flat fill.
        fillMode: ((fill instanceof Map) ? fill.get("mode") : null) ?: "flat",
    ]
}

def appendPublishedBeadRemovalChecks(errors, row, settings, params) {
    // With the build stage skipped (start/only stage past BUILD_SPATIALDATA) the
    // published source zarr feeds the later stages without passing the Python
    // reuse guard, so a mismatch with the requested bead removal is refused here:
    // it must never run silently on the other image.
    if (!settings.need_build_results) {
        return
    }
    def sourceZarr = normalizedPath(
        publishedDatasetPath(
            params.outdir,
            settings.pair_id,
            "MERSCOPE",
            "spatialdata/spatialdata_out/source_spatialdata.zarr",
        )
    )
    if (!sourceZarr.toFile().isDirectory()) {
        // Reported when the path is required (requireExistingPath).
        return
    }
    def requested = merscopeBeadRemovalRequested(row, settings.pair_id, params)
    def requestedFill = requested && merscopeBeadRemovalFillRequested(params)
    def requestedFillMode = merscopeBeadRemovalFillMode(params)
    def recorded = publishedBeadRemovalState(sourceZarr)
    def remedy = (
        "Run from the build stage with --force_spatialdata_build true (needs " +
        "merscope_dir) or change merscope_bead_removal / " +
        "merscope_bead_removal_fill / merscope_bead_removal_fill_mode to match " +
        "the published zarr."
    )
    if (recorded.enabled != requested) {
        errors << (
            "The published MERSCOPE source zarr ${sourceZarr} was built with " +
            "bead removal ${onOffText(recorded.enabled)} but ${onOffText(requested)} " +
            "is requested. ${remedy}"
        )
    } else if (requested && recorded.fill != requestedFill) {
        errors << (
            "The published MERSCOPE source zarr ${sourceZarr} was built with " +
            "the bead-removal fill ${onOffText(recorded.fill)} but " +
            "${onOffText(requestedFill)} is requested. ${remedy}"
        )
    } else if (requestedFill && recorded.fillMode != requestedFillMode) {
        errors << (
            "The published MERSCOPE source zarr ${sourceZarr} was built with " +
            "the bead-removal fill mode ${recorded.fillMode} but " +
            "${requestedFillMode} is requested. ${remedy}"
        )
    }
}

def appendGastonPreflightChecks(errors, settings, params) {
    if (!(settings.run_gaston && settings.gaston_published_output_mode)) {
        return
    }
    settings.gaston_segmentations.each { segmentation ->
        settings.active_platforms.each { platform ->
            def sampleId = "${settings.pair_id}_${platform}"
            def clusteredH5ad = normalizedPath(
                publishedPairPath(
                    params.outdir,
                    settings.pair_id,
                    "${segmentation}/" +
                    AnnotationSettings.publishedClusteringDir(
                        params.outdir, settings.pair_id, segmentation, platform, sampleId,
                        settings.clustering_squidpy_table_key_suffix, // rca-site:H2
                    ) +
                    "/clustering_squidpy_out/${platform.toLowerCase()}/" +
                    "${sampleId}_clustered.h5ad",
                )
            )
            if (!clusteredH5ad.toFile().isFile()) {
                errors << (
                    "Missing GASTON clustered H5AD for " +
                    "${settings.pair_id}:${platform}:${segmentation}: ${clusteredH5ad}"
                )
            }
            def latestZarr = normalizedPath(
                publishedDatasetPath(
                    params.outdir,
                    settings.pair_id,
                    platform,
                    "latest/latest_spatialdata.zarr",
                )
            )
            if (!latestZarr.toFile().isDirectory()) {
                errors << (
                    "Missing GASTON latest SpatialData Zarr for " +
                    "${settings.pair_id}:${platform}:${segmentation}: ${latestZarr}"
                )
            } else {
                def layerKeys = analysisLayerKeys(platform, segmentation)
                def tableKey = clusteredSpatialdataTableKey(
                    layerKeys.table_key,
                    segmentation,
                    settings.clustering_squidpy_table_key_suffix, // rca-site:H2
                )
                def tablePath = latestZarr.resolve("tables").resolve(tableKey)
                if (!tablePath.toFile().exists()) {
                    errors << (
                        "Missing GASTON clustered SpatialData table '${tableKey}' " +
                        "for ${settings.pair_id}:${platform}:${segmentation}: " +
                        "${tablePath}"
                    )
                }
            }
        }
    }
}

def pciseqPublishedH5adPath(pairId, segmentation, platform, params) {
    return normalizedPath(publishedPairPath(
        params.outdir,
        pairId,
        "${segmentation}/clustering_squidpy/clustering_squidpy_out/" +
            "${platform.toLowerCase()}/${pairId}_${platform}_clustered.h5ad",
    ))
}

def pciseqSpatialdataPath(row, settings, platform, params) {
    def fields = PlatformSpec.forName(platform).spatialdataPathFields
    return normalizedPath(chooseField(row, fields) ?: publishedDatasetPath(
        params.outdir,
        settings.pair_id,
        platform,
        "latest/latest_spatialdata.zarr",
    ))
}

def normalizePciseqRegionLabels(rawValue) {
    if (rawValue == null) {
        return []
    }
    def values = rawValue
    if (!(values instanceof Collection)) {
        def raw = values.toString().trim()
        values = raw.startsWith("[")
            ? new groovy.json.JsonSlurper().parseText(raw)
            : raw.split(",").toList()
    }
    if (!(values instanceof Collection)) {
        throw new IllegalArgumentException("pciseq_region_labels must be a list")
    }
    return values.collect { value -> value.toString().trim() }
        .findAll { value -> value }
        .unique()
        .sort()
}

def pciseqReferenceSettings(row, species, params) {
    def referencePath = optionalNormalizedPathString(
        rowFieldOrDefault(row, "pciseq_reference_path", params.pciseq_reference_path),
    )
    def source = rowFieldOrDefault(
        row, "pciseq_reference_source", params.pciseq_reference_source,
    ).toString().trim().toLowerCase().replaceAll(/[-_]/, "")
    // A supplied prebuilt matrix retains the original pciSeq input contract.
    if (referencePath != null && source == "auto") {
        source = "custom"
    } else if (source == "auto") {
        source = species == "mouse" ? "wmb" : "whb"
    }
    def regionLabels = normalizePciseqRegionLabels(rowFieldOrDefault(
        row, "pciseq_region_labels", params.pciseq_region_labels,
    ))
    def regionMode = rowFieldOrDefault(
        row, "pciseq_region_mode", params.pciseq_region_mode,
    ).toString().trim().toLowerCase()
    if (regionMode == "auto") {
        regionMode = regionLabels || species == "human" ? "required" : "whole_brain"
    }
    // TODO: after coronal-slice registration to Allen CCFv3, restrict mouse
    // classes to intersected regions plus a 100 µm anatomical safety buffer.
    def referenceLevel = rowFieldOrDefault(
        row, "pciseq_reference_level", params.pciseq_reference_level,
    ).toString().trim().toLowerCase()
    if (referenceLevel == "auto" && source in ["wmb", "whb", "seaad"]) {
        referenceLevel = source == "whb" ? "cluster" : "subclass"
    }
    return [
        species: species,
        source: source,
        reference_root: optionalNormalizedPathString(rowFieldOrDefault(
            row, "pciseq_reference_root", params.pciseq_reference_root,
        )),
        reference_path: referencePath,
        region_labels: regionLabels,
        region_mode: regionMode,
        level: referenceLevel,
        min_cells_per_type: params.pciseq_reference_min_cells as int,
        normalization_target_sum: params.pciseq_reference_normalize_target_sum as double,
        donor_balance: boolOrDefault(
            params.pciseq_reference_donor_balance, true, "pciseq_reference_donor_balance",
        ),
        seaad_cohort: rowFieldOrDefault(
            row, "pciseq_seaad_cohort", params.pciseq_seaad_cohort,
        ).toString().trim().toLowerCase(),
        auto_download: boolOrDefault(
            params.pciseq_reference_auto_download, true, "pciseq_reference_auto_download",
        ),
        verify_checksums: boolOrDefault(
            params.pciseq_reference_verify_checksums, true, "pciseq_reference_verify_checksums",
        ),
        gene_id_mode: params.pciseq_reference_gene_id_mode.toString(),
        custom_metadata_path: optionalNormalizedPathString(rowFieldOrDefault(
            row, "pciseq_reference_custom_metadata_path", params.pciseq_reference_custom_metadata_path,
        )),
        output_dir: "pciseq_reference_out",
    ]
}

def pciseqReferenceKey(referenceSettings) {
    def digest = java.security.MessageDigest.getInstance("SHA-256")
    def json = groovy.json.JsonOutput.toJson(referenceSettings)
    return digest.digest(json.getBytes("UTF-8")).encodeHex().toString()
}

def appendPciseqPreflightChecks(errors, row, settings, params) {
    if (!settings.run_pciseq) {
        return
    }
    def reference = settings.pciseq_reference
    if (!(reference.source in ["wmb", "whb", "seaad", "custom"])) {
        errors << "pciseq_reference_source must be auto, wmb, whb, seaad, or custom"
    }
    if ((reference.species == "mouse" && reference.source in ["whb", "seaad"]) ||
        (reference.species == "human" && reference.source == "wmb")) {
        errors << "PCISEQ reference source '${reference.source}' is incompatible with species '${reference.species}'"
    }
    if (!(reference.region_mode in ["required", "whole_brain"])) {
        errors << "pciseq_region_mode must be auto, required, or whole_brain"
    }
    if (reference.region_mode == "required" && !reference.region_labels) {
        errors << (
            "PCISEQ requires pciseq_region_labels for ${settings.pair_id}; " +
            "set pciseq_region_mode=whole_brain explicitly to use an unrestricted reference"
        )
    }
    if (reference.region_mode == "whole_brain" && reference.region_labels) {
        errors << "PCISEQ whole_brain mode requires empty pciseq_region_labels"
    }
    if (!(reference.seaad_cohort in ["all", "ad", "control"])) {
        errors << "pciseq_seaad_cohort must be all, ad, or control"
    }
    if (reference.source != "seaad" && reference.seaad_cohort != "all") {
        errors << "pciseq_seaad_cohort applies only to the seaad reference source"
    }
    if (reference.source == "custom") {
        appendPreflightFileCheck(
            errors,
            reference.reference_path,
            "PCISEQ reference expression matrix for ${settings.pair_id}",
        )
        if (reference.region_labels && isBlankPath(reference.custom_metadata_path)) {
            errors << "PCISEQ custom reference region filtering requires pciseq_reference_custom_metadata_path"
        }
    } else if (!isBlankPath(reference.reference_path)) {
        errors << "PCISEQ pciseq_reference_path requires pciseq_reference_source=custom or auto"
    }
    if (!isBlankPath(reference.custom_metadata_path)) {
        appendPreflightFileCheck(
            errors, reference.custom_metadata_path, "PCISEQ custom reference metadata",
        )
    }
    if (!(params.pciseq_geometry_mode in ["project_2d", "labels_3d"])) {
        errors << "pciseq_geometry_mode must be project_2d or labels_3d"
    }
    if (params.pciseq_geometry_mode == "labels_3d") {
        appendPreflightFileCheck(errors, params.pciseq_labels_path, "PCISEQ 3D labels")
    }
    if (settings.run_clustering_squidpy) {
        return
    }
    settings.analysis_segmentations.each { segmentation ->
        settings.active_platforms.each { platform ->
            def label = "${settings.pair_id}:${platform}:${segmentation}"
            def h5ad = pciseqPublishedH5adPath(
                settings.pair_id, segmentation, platform, params,
            )
            if (!h5ad.toFile().isFile()) {
                errors << "Missing PCISEQ clustered H5AD for ${label}: ${h5ad}"
            }
            def zarr = pciseqSpatialdataPath(row, settings, platform, params)
            if (!zarr.toFile().isDirectory()) {
                errors << "Missing PCISEQ latest SpatialData Zarr for ${label}: ${zarr}"
                return
            }
            def keys = analysisLayerKeys(platform, segmentation)
            def elements = [["tables", keys.table_key]]
            if (params.pciseq_geometry_mode == "project_2d") {
                elements << ["shapes", keys.shape_key]
            }
            if (!isBlankPath(params.pciseq_points_key)) {
                elements << ["points", params.pciseq_points_key.toString()]
            }
            elements.each { kind, key ->
                def element = zarr.resolve(kind).resolve(key)
                if (!element.toFile().exists()) {
                    errors << "Missing PCISEQ native ${kind} '${key}' for ${label}: ${element}"
                }
            }
        }
    }
}

def pciseqConfigJson(pairId, segmentation, platform, sampleId, spatialdataPath, settings, params) {
    def keys = analysisLayerKeys(platform, segmentation)
    def parseArray = { value ->
        return value instanceof Collection
            ? value
            : new groovy.json.JsonSlurper().parseText(value.toString())
    }
    def options = params.pciseq_options instanceof Map
        ? params.pciseq_options
        : new groovy.json.JsonSlurper().parseText(params.pciseq_options.toString())
    return groovy.json.JsonOutput.prettyPrint(groovy.json.JsonOutput.toJson([
        pair_id: pairId,
        sample_id: sampleId,
        platform: platform,
        segmentation: segmentation,
        source_h5ad: "source_clustered.h5ad",
        spatialdata_path: spatialdataPath.toString(),
        reference_path: "reference_input/reference.csv",
        reference_manifest_path: "reference_input/reference_manifest.json",
        table_key: keys.table_key,
        shape_key: keys.shape_key,
        points_key: params.pciseq_points_key,
        geometry_mode: params.pciseq_geometry_mode,
        overlap_policy: params.pciseq_overlap_policy,
        pixel_size_um: params.pciseq_pixel_size_um,
        max_pixels: params.pciseq_max_pixels,
        coordinate_system: params.pciseq_coordinate_system,
        labels_path: optionalNormalizedPathString(params.pciseq_labels_path),
        origin_um: parseArray.call(params.pciseq_origin_um),
        voxel_size_um: parseArray.call(params.pciseq_voxel_size_um),
        max_iter: params.pciseq_max_iter,
        n_neighbors: params.pciseq_n_neighbors,
        options: options,
        write_spatialdata_table: boolOrDefault(
            params.pciseq_write_spatialdata_table, true, "pciseq_write_spatialdata_table",
        ),
        output_dir: "pciseq_out",
    ]))
}

// Stages an ATERA row may select with the native segmentation route (plan
// AT-6). The others are refused with one message each below.
def ateraNativeStages() {
    return [
        "build_spatialdata",
        "train_cellpose",
        "segment",
        "enrich",
        "build_viewer_caches",
        "mask_image_quantification",
        "qc",
        "visualize",
        "spatial_gene_analysis",
        "mecr",
        "clustering_squidpy",
        "mender",
    ]
}

// Whether a samplesheet row analyses an ATERA section (its analysis_mode, else
// --analysis_mode). An invalid mode answers false here; rowSampleSettings then
// reports it.
def rowIsAtera(row, params) {
    try {
        def analysisMode = normalizeAnalysisMode(
            rowFieldOrDefault(row, "analysis_mode", params.analysis_mode)
        )
        return activePlatformsForMode(analysisMode).contains("ATERA")
    } catch (IllegalArgumentException ignored) {
        return false
    }
}

// ATERA keeps its legacy task hashes unless a bound hippocampal resource
// manifest opts it into RCA. That profile has its own taxonomy and confidence
// evidence; it never inherits the targeted frontal-panel rules.
def annotationModeRow(row, params) {
    if (!rowIsAtera(row, params)) {
        return row
    }
    if (params.annotation_hippocampal_resources || chooseField(row, [AnnotationDefaults.ROW_MODE_COLUMN]) != null) {
        return row
    }
    return row + [(AnnotationDefaults.ROW_MODE_COLUMN): AnnotationDefaults.LEGACY]
}

// The launch-log lines of the run's annotation modes, logged once before any
// channel runs (hook H5 site):
// - info: the per-row mode summary, where an ATERA row held in legacy mode by
//   annotationModeRow reads "legacy (ATERA)" rather than "(row)", and the note
//   naming those rows;
// - warn: the legacy deprecation warning of a flipped species. It leaves out
//   every ATERA row: holding one in legacy mode, or a row that names legacy as
//   the ATERA refusal advises, is not a deprecated choice.
def annotationModeLaunchLog(samplesheetRows, params) {
    def ateraRows = samplesheetRows.findAll { row -> rowIsAtera(row, params) }
    def ateraLegacyRows = ateraRows.findAll { row ->
        !params.annotation_hippocampal_resources && chooseField(row, [AnnotationDefaults.ROW_MODE_COLUMN]) == null
    }
    def info = []
    def summary = AnnotationSettings.rowModeSummary(
        params,
        samplesheetRows.collect { row -> annotationModeRow(row, params) },
        AnnotationDefaults.FLIPPED_SPECIES,
        ateraLegacyRows.collect { row -> row.pair_id },
    )
    if (summary) {
        info.add(summary)
    }
    if (ateraLegacyRows) {
        info.add(
            "ATERA rows ${ateraLegacyRows.collect { row -> row.pair_id }.join(', ')} " +
            "cluster in legacy mode: map_first annotation is not covered for ATERA " +
            "rows yet (AT-11, RCA on Atera)"
        )
    }
    def warn = []
    def deprecation = AnnotationSettings.deprecatedLegacyWarning(
        params,
        samplesheetRows
            .findAll { row -> !rowIsAtera(row, params) }
            .collect { row -> row[AnnotationDefaults.ROW_MODE_COLUMN] },
    )
    if (deprecation) {
        warn.add(deprecation)
    }
    return [info: info.collect { line -> line.toString() }, warn: warn]
}

// Whether --annotation_prepare_only leaves a (pair_id, row, settings) entry out:
// an ATERA row builds no reference bundle, as its rows cluster in legacy mode
// (AT-11, RCA on Atera; annotationModeRow).
def ateraPrepareOnlyRow(entry) {
    def settings = entry[2]
    if (!("ATERA" in settings.active_platforms)) {
        return false
    }
    if (params.annotation_hippocampal_resources) {
        log.info("annotation_prepare_only: ATERA row ${settings.pair_id} validates the bound hippocampal references separately")
        return true
    }
    log.info(
        "annotation_prepare_only: ATERA row ${settings.pair_id} builds no reference " +
        "bundle: map_first annotation is not covered for ATERA rows yet (AT-11, RCA on Atera)"
    )
    return true
}

def appendAteraPreflightChecks(errors, row, settings, params) {
    if (!("ATERA" in settings.active_platforms)) {
        return
    }
    if (!settings.selected_stages) {
        errors.add(
            "ATERA row ${settings.pair_id} selects no stage; ATERA rows run " +
            "${ateraNativeStages().join(', ')}"
        )
        return
    }
    def stageOrder = settings.stage_order
    appendAteraProsegPreflightChecks(errors, row, settings, params)
    if (settings.run_mapmycells) {
        errors.add(
            "standalone MapMyCells is not covered for ATERA rows; stop at " +
            "clustering_squidpy or mender"
        )
    }
    if (AnnotationSettings.isMapFirst(settings) && AnnotationPreflight.usesClusteredTables(settings)) {
        if (!params.annotation_hippocampal_resources) {
            errors.add("map_first annotation requires --annotation_hippocampal_resources for ATERA rows (AT-11)")
        }
        if (settings.species != "human" || settings.annotation_anatomical_region != "hippocampus") {
            errors.add("ATERA map_first requires human species and explicit anatomical_region=hippocampus")
        }
    }

    if (stageOrder.contains("gaston")) {
        errors.add(
            "GASTON is not covered for ATERA rows; remove gaston_enabled from " +
            "row ${settings.pair_id} or set it to false"
        )
    }
    if (stageOrder.contains("pciseq")) {
        errors.add(
            "pciSeq is off for ATERA rows; remove pciseq_enabled from row " +
            "${settings.pair_id} or set it to false"
        )
    }
    if (
        stageOrder.contains("compute_cortical_depth") ||
        stageOrder.contains("distance_from_object")
    ) {
        errors.add(
            "cortical depth and distance from object are off for non-cortical " +
            "ATERA sections; remove cortical_depth_enabled and " +
            "distance_from_object_enabled from row ${settings.pair_id} or set " +
            "them to false"
        )
    }
    def alignmentRequested = boolOrDefault(
        chooseField(row, ["enable_alignment"]),
        false,
        "enable_alignment for ${settings.pair_id}",
    )
    if (
        alignmentRequested ||
        chooseField(row, ["xenium_registered_images_csv"]) != null
    ) {
        errors.add(
            "image registration and alignment are not covered for ATERA rows (AT-10)"
        )
    }
    def unsupported = settings.selected_stages.findAll { stage ->
        !(stage in ateraNativeStages()) && stage != "mapmycells"
    }
    if (unsupported) {
        errors.add(
            "ATERA rows run only ${ateraNativeStages().join(', ')}; " +
            "${settings.pair_id} also selects ${unsupported.join(', ')}"
        )
    }
}

// NATIVE_LATEST config of a native row: the latest store is built at the
// persistent latest path from the staged source store.
def nativeLatestConfigForPlatform(pairId, platform, params) {
    return [
        dataset_name: "${pairId}_${platform}",
        platform: platform,
        source_zarr_path: "native_source.zarr",
        output_dir: "native_out",
        persistent_latest_zarr_path: file(
            publishedDatasetPath(
                params.outdir,
                pairId,
                platform,
                "latest/latest_spatialdata.zarr",
            )
        ).toAbsolutePath().toString(),
        link_mode: params.native_latest_link_mode.toString().trim().toLowerCase(),
    ]
}

// A published NATIVE_LATEST output (restart from a later stage).
def publishedNativeOutput(params, pairId, platform, name) {
    return requireExistingPath(
        publishedDatasetPath(
            params.outdir,
            pairId,
            platform,
            "segmentation/native_out/${name}",
        ),
        "NATIVE_LATEST ${name} for ${pairId}:${platform}",
    )
}

// ---------------------------------------------------------------------------
// ATERA ProSeg route (segmentation_source=native_proseg, plan AT-9)
// ---------------------------------------------------------------------------

// A numeric ATERA ProSeg param; command-line values arrive as strings.
def ateraProsegNumber(params, name, integer) {
    def raw = params[name]
    if (raw == null || raw instanceof Boolean || !raw.toString().trim()) {
        throw new IllegalArgumentException(
            "${name} must be set for segmentation_source=native_proseg rows"
        )
    }
    def value
    try {
        value = new BigDecimal(raw.toString().trim())
    } catch (NumberFormatException error) {
        throw new IllegalArgumentException("${name}=${raw} is not a number")
    }
    if (integer) {
        try {
            return value.intValueExact()
        } catch (ArithmeticException error) {
            throw new IllegalArgumentException("${name}=${raw} is not an integer")
        }
    }
    return value.doubleValue()
}

// Tile ids of atera_proseg_tile_subset (a check run), or null for every tile.
def ateraProsegTileSubset(params) {
    def raw = params.atera_proseg_tile_subset
    if (raw == null || raw instanceof Boolean || !raw.toString().trim()) {
        return null
    }
    def ids = raw.toString().split(",").collect { value -> value.trim() }.findAll { value ->
        value.length() > 0
    }
    return ids ? ids : null
}

// ProSeg model and schedule of every tile (AteraProsegModelParams). The
// Cellpose prior (AT-7) takes its effective prior weight from the plan (the
// vendor one stays atera_proseg_prior_weight) and its own nuclear-compartment
// switch (off by default: no vendor segmentation feeds the Cellpose route, as
// the XENIUM CSV route has no compartment).
def ateraProsegModelParams(params, priorKind = "vendor") {
    def model = ateraProsegVendorModelParams(params)
    if (priorKind == "cellpose") {
        model.prior_weight_effective = null
        model.nuclear_compartment = boolOrDefault(
            params.atera_cellpose_proseg_nuclear_compartment,
            false,
            "atera_cellpose_proseg_nuclear_compartment",
        )
    }
    return model
}

def ateraProsegVendorModelParams(params) {
    return [
        voxel_size: ateraProsegNumber(params, "proseg_voxel_size", false),
        burnin_voxel_size: ateraProsegNumber(params, "proseg_burnin_voxel_size", false),
        voxel_layers: ateraProsegNumber(params, "default_atera_voxel_layers", true),
        nuclear_reassignment_prob: ateraProsegNumber(
            params, "proseg_nuclear_reassignment_prob", false
        ),
        diffusion_probability: ateraProsegNumber(
            params, "proseg_diffusion_probability", false
        ),
        cell_compactness: ateraProsegNumber(params, "proseg_cell_compactness", false),
        burnin_samples: ateraProsegNumber(params, "atera_proseg_burnin_samples", true),
        samples: ateraProsegNumber(params, "atera_proseg_samples", true),
        hillclimb: ateraProsegNumber(params, "atera_proseg_hillclimb", true),
        recorded_samples: ateraProsegNumber(params, "atera_proseg_recorded_samples", true),
        prior_weight_effective: ateraProsegNumber(params, "atera_proseg_prior_weight", false),
        min_qv: params.atera_min_qv == null ? 0.0d : ateraProsegNumber(params, "atera_min_qv", false),
        nuclear_compartment: boolOrDefault(
            params.atera_proseg_nuclear_compartment,
            true,
            "atera_proseg_nuclear_compartment",
        ),
    ]
}

// Refusals of a tiled ATERA ProSeg row (native_proseg, AT-9; ATERA
// cellpose_proseg, AT-7) that the Pydantic models would only raise inside a
// task (plan AT-9 §2.2).
def appendAteraProsegPreflightChecks(errors, row, settings, params) {
    if (!runsAteraProseg(settings.segmentation_source, settings.active_platforms)) {
        return
    }
    def cellposePrior = settings.segmentation_source in ["cellpose_proseg", "cellpose_comparison"]
    def checks = []
    if (cellposePrior) {
        def channels = parseChannels(row.atera_channels, [])
        if (row.atera_channels != null && !channels) {
            checks << "atera_channels of ${settings.pair_id} names no channel"
        }
        def weight = params.atera_cellpose_proseg_prior_weight
        if (weight != null && !(weight instanceof Boolean) && weight.toString().trim()) {
            try {
                if (ateraProsegNumber(params, "atera_cellpose_proseg_prior_weight", false) <= 0) {
                    checks << "atera_cellpose_proseg_prior_weight must be positive"
                }
            } catch (IllegalArgumentException error) {
                checks << error.getMessage()
            }
        }
        ["atera_cellpose_proseg_nuclear_compartment", "atera_cellpose_proseg_cellprobs"].each {
            name ->
                try {
                    boolOrDefault(params[name], true, name)
                } catch (IllegalArgumentException error) {
                    checks << error.getMessage()
                }
        }
    }
    try {
        def halo = ateraProsegNumber(params, "atera_proseg_halo_um", false)
        def tile = ateraProsegNumber(params, "atera_proseg_tile_um", false)
        if (halo < 100) {
            checks << "atera_proseg_halo_um=${halo} is below the minimum of 100 microns"
        }
        if (tile <= 2 * halo) {
            checks << "atera_proseg_tile_um=${tile} must exceed twice the halo (${halo})"
        }
        if (ateraProsegNumber(params, "atera_proseg_owned_margin_um", false) < 0) {
            checks << "atera_proseg_owned_margin_um must be at least 0"
        }
        def model = ateraProsegModelParams(params)
        ["burnin_samples", "hillclimb", "recorded_samples"].each { name ->
            if (model[name] < 0) {
                checks << "atera_proseg_${name} must be at least 0"
            }
        }
        if (model.samples < 1) {
            checks << "atera_proseg_samples must be at least 1"
        }
        if (model.recorded_samples > model.samples) {
            checks << (
                "atera_proseg_recorded_samples=${model.recorded_samples} exceeds " +
                "atera_proseg_samples=${model.samples}"
            )
        }
        if (!cellposePrior && model.prior_weight_effective <= 0) {
            checks << "atera_proseg_prior_weight must be positive"
        }
        if (ateraProsegNumber(params, "atera_proseg_tile_cpus", true) < 1) {
            checks << "atera_proseg_tile_cpus must be at least 1"
        }
    } catch (IllegalArgumentException error) {
        checks << error.getMessage()
    }
    def allowUntested = boolOrDefault(
        params.atera_proseg_allow_untested_version,
        false,
        "atera_proseg_allow_untested_version",
    )
    if (params.proseg_version?.toString()?.trim() != "3.2.0" && !allowUntested) {
        checks << (
            "the ATERA ProSeg route is verified with ProSeg 3.2.0 only (Xenium-layout " +
            "parquet reader and flags); proseg_version=${params.proseg_version}. Set " +
            "atera_proseg_allow_untested_version=true to run it anyway"
        )
    }
    def subset = ateraProsegTileSubset(params)
    if (subset != null) {
        def order = settings.stage_order
        def later = settings.selected_stages.findAll { stage ->
            order.indexOf(stage) > order.indexOf("segment")
        }
        if (later) {
            checks << (
                "atera_proseg_tile_subset (${subset.join(',')}) is a check run of " +
                "single tiles and cannot be stitched; ${settings.pair_id} also " +
                "selects ${later.join(', ')}. Use stop_stage=segment"
            )
        }
    }
    errors.addAll(checks)
}

// ATERA_PROSEG_PLAN config: the plan reads the staged source store. A
// Cellpose plan (AT-7) also reads the staged Cellpose mask and logits and fixes
// the effective prior weight of every tile (null: the whole-section automatic
// weight).
def ateraProsegPlanConfig(pairId, platform, params, priorKind = "vendor") {
    def config = [
        dataset_name: "${pairId}_${platform}",
        platform: platform,
        source_zarr_path: "proseg_source.zarr",
        output_dir: "plan_out",
        tile_um: ateraProsegNumber(params, "atera_proseg_tile_um", false),
        halo_um: ateraProsegNumber(params, "atera_proseg_halo_um", false),
        owned_margin_um: ateraProsegNumber(params, "atera_proseg_owned_margin_um", false),
    ]
    def subset = ateraProsegTileSubset(params)
    if (subset != null) {
        config.tile_subset = subset
    }
    // Only for a Cellpose plan, so the vendor plan config (and its task hash)
    // is unchanged.
    if (priorKind == "cellpose") {
        def cellprobs = boolOrDefault(
            params.atera_cellpose_proseg_cellprobs,
            true,
            "atera_cellpose_proseg_cellprobs",
        )
        config.prior = [
            kind: "cellpose",
            mask_path: "prior_mask.npy",
            cellprob_path: cellprobs ? "prior_cellprob.npy" : null,
        ]
        def weight = params.atera_cellpose_proseg_prior_weight
        config.prior_weight_effective = (
            weight == null || weight instanceof Boolean || !weight.toString().trim()
        ) ? null : ateraProsegNumber(params, "atera_cellpose_proseg_prior_weight", false)
    }
    return config
}

// The tiles a plan runs: one [tile_id, AteraProsegTileConfig map] per tile.
// The config is a value (not the plan file), so a re-run plan with the same
// content does not invalidate finished tiles.
def ateraProsegTileSpecs(planText, pairId, platform, sourceZarrPath, params) {
    def plan = new groovy.json.JsonSlurper().parseText(planText)
    def priorKind = plan.prior ? plan.prior.kind.toString() : "vendor"
    def model = ateraProsegModelParams(params, priorKind)
    if (priorKind == "cellpose") {
        model.prior_weight_effective = plan.prior_weight.effective as double
    }
    def gitRev = params.proseg_git_rev?.toString()?.trim()
    return plan.tiles.findAll { tile -> tile.run }.collect { tile ->
        def config = [
            dataset_name: "${pairId}_${platform}",
            platform: platform,
            tile_id: tile.tile_id,
            source_zarr_path: sourceZarrPath,
            source_build_fingerprint: plan.source_build_fingerprint,
            pixel_size_um: plan.pixel_size_um,
            core_um: tile.core_um,
            window_px: tile.window_px,
            mask_shape_px: plan.mask_shape_px,
            owned_labels: tile.owned_labels,
            output_dir: "tile_${tile.tile_id}",
            num_threads: ateraProsegNumber(params, "atera_proseg_tile_cpus", true),
            max_rss_gb: ateraProsegNumber(params, "atera_proseg_tile_max_rss_gb", false),
            model: model,
            proseg_git_rev: gitRev ? gitRev : null,
            keep_inputs: boolOrDefault(
                params.atera_proseg_keep_tile_inputs,
                false,
                "atera_proseg_keep_tile_inputs",
            ),
        ]
        // Only when set, so the default config (and the task hash of
        // finished tiles) is unchanged.
        if (boolOrDefault(
            params.atera_proseg_allow_untested_version,
            false,
            "atera_proseg_allow_untested_version",
        )) {
            config.allow_untested_version = true
        }
        // The Cellpose prior (AT-7): the files the plan fingerprinted, by real
        // path; a vendor tile config has no entry (its task hash is unchanged).
        if (priorKind == "cellpose") {
            config.prior = [
                kind: "cellpose",
                mask_path: plan.prior.mask_path,
                cellprob_path: plan.prior.cellprob_path,
                mask_stat: plan.prior.mask_stat,
                cellprob_stat: plan.prior.cellprob_stat,
                fingerprint: plan.prior.fingerprint,
            ]
        }
        [tile.tile_id.toString(), config]
    }
}

// ATERA_PROSEG_STITCH config: the stitch writes the ProSeg layer of the prior
// kind (proseg_vendor or, for a Cellpose prior, reseg) into the persistent
// latest store of the row. prior_kind is always set: it entered the config with
// the AT-7 rename of the vendor layer, so stitches made before it re-run.
def ateraProsegStitchConfig(pairId, platform, tileIds, params, priorKind = "vendor") {
    return [
        dataset_name: "${pairId}_${platform}",
        platform: platform,
        latest_zarr_path: "native_latest.zarr",
        persistent_latest_zarr_path: file(
            publishedDatasetPath(
                params.outdir,
                pairId,
                platform,
                "latest/latest_spatialdata.zarr",
            )
        ).toAbsolutePath().toString(),
        plan_path: "tile_plan.json",
        tile_dirs: tileIds.collect { tileId -> "tile_${tileId}".toString() }.sort(),
        output_dir: "stitch_out",
        prior_kind: priorKind,
    ]
}

// The published ATERA Cellpose mask of a cellpose_proseg row (restart from a
// later stage).
def publishedAteraCellposeMask(params, pairId, platform) {
    return requireExistingPath(
        publishedDatasetPath(
            params.outdir,
            pairId,
            platform,
            "segmentation/cellpose_masks_tiled.npy",
        ),
        "ATERA_CELLPOSE_SEGMENT Cellpose mask for ${pairId}:${platform}",
    )
}

// A published ATERA_PROSEG_STITCH output (restart from a later stage).
def publishedAteraProsegOutput(params, pairId, platform, name) {
    return requireExistingPath(
        publishedDatasetPath(
            params.outdir,
            pairId,
            platform,
            "segmentation/stitch_out/${name}",
        ),
        "ATERA_PROSEG_STITCH ${name} for ${pairId}:${platform}",
    )
}

def runPreflightChecks(row, settings, params) {
    def ateraErrors = []
    appendAteraPreflightChecks(ateraErrors, row, settings, params)
    if (ateraErrors) {
        throw new IllegalArgumentException(
            "Preflight checks failed for sample ${settings.pair_id}:\n" +
            ateraErrors.collect { errorMessage -> " - ${errorMessage}" }.join("\n")
        )
    }
    CellposeSettings.preflight(row, settings, params)
    def errors = []
    appendRegisterImagesPreflightChecks(errors, settings)
    appendClusteringSquidpyPreflightChecks(errors, settings, params)
    appendMapMyCellsPreflightChecks(errors, settings, params)
    appendAlignmentAnnotationPreflightChecks(errors, row, settings, params)
    appendCorticalDepthPreflightChecks(errors, row, settings, params)
    appendSpatialGeneTranscriptPreflightChecks(errors, row, settings, params)
    appendDistanceFromObjectPreflightChecks(errors, row, settings, params)
    appendMerscopeBeadRemovalPreflightChecks(errors, row, settings, params)
    appendMecrPreflightChecks(errors, settings, params)
    appendMenderPreflightChecks(errors, settings, params)
    appendGastonPreflightChecks(errors, settings, params)
    appendPciseqPreflightChecks(errors, row, settings, params)
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

def merscopeBeadRemovalRequested(row, pairId, params) {
    return boolOrDefault(
        rowFieldOrDefault(row, "merscope_bead_removal", params.merscope_bead_removal),
        false,
        "merscope_bead_removal for ${pairId}"
    )
}

def merscopeBeadRemovalDevice(params) {
    def rawDevice = params.merscope_bead_removal_device
    if (rawDevice == null || rawDevice.toString().trim().isEmpty()) {
        return "cpu"
    }
    return rawDevice.toString().trim().toLowerCase()
}

// The residual-footprint fill is on by default (hybrid mode); false switches it
// off for every sample that has bead removal on.
def merscopeBeadRemovalFillRequested(params) {
    return boolOrDefault(
        params.merscope_bead_removal_fill,
        true,
        "merscope_bead_removal_fill"
    )
}

// The veto of Cellpose cell masks on modelled beads is on by default; false
// switches it off. It only reaches the config of samples with bead removal on.
def merscopeBeadMaskVetoRequested(params) {
    return boolOrDefault(
        params.merscope_bead_mask_veto,
        true,
        "merscope_bead_mask_veto"
    )
}

def merscopeBeadRemovalFillMode(params) {
    def rawMode = params.merscope_bead_removal_fill_mode
    if (rawMode == null || rawMode.toString().trim().isEmpty()) {
        return "hybrid"
    }
    return rawMode.toString().trim().toLowerCase()
}

// A global switch of a stage that is off for ATERA rows is ignored on them with
// a setting note, so a mixed samplesheet keeps it for its other rows. A value
// set on the ATERA row itself is kept, and the ATERA preflight refuses it.
def ateraIgnoresGlobalSwitch(enabled, ateraRow, row, fieldName, stageLabel, settingNotes) {
    if (!ateraRow || chooseField(row, [fieldName]) != null) {
        return enabled
    }
    if (enabled) {
        settingNotes << (
            "${stageLabel} is off for ATERA rows; the global ${fieldName} is ignored"
        ).toString()
    }
    return false
}

def rowSampleSettings(row, params) {
    def pairId = row.pair_id?.toString()?.trim()
    if (!pairId) {
        error "Found samplesheet row with missing pair_id: ${row}"
    }

    def species = normalizeSpecies(params.species)
    def analysisMode = normalizeAnalysisMode(
        rowFieldOrDefault(row, "analysis_mode", params.analysis_mode)
    )
    def pairedMode = analysisMode == "paired"
    def activePlatforms = activePlatformsForMode(analysisMode)
    def segmentationSource = segmentationSourceForRow(row, activePlatforms, params)
    // Every ATERA route builds its latest store with NATIVE_LATEST (AT-7).
    def nativeSegmentation = usesNativeLatest(segmentationSource, activePlatforms)
    // Whole-transcriptome ATERA rows use the label-raster transcript patterns
    // and their own streaming MECR reference (MECR_REFERENCE_ATERA).
    def ateraRow = activePlatforms.contains("ATERA")
    // Settings resolved differently from what was asked, logged per row.
    def settingNotes = []
    def rowAnalysisSegmentation = chooseField(row, ["analysis_segmentation"])
    def analysisSegmentations = resolveRowSegmentations(
        rowAnalysisSegmentation,
        rowFieldOrDefault(row, "analysis_segmentation", params.analysis_segmentation),
        { value -> normalizeAnalysisSegmentation(value, segmentationSource) },
        segmentationSource,
        "analysis_segmentation",
        pairId,
        settingNotes,
        activePlatforms,
    )
    def pciseqEnabled = ateraIgnoresGlobalSwitch(
        boolOrDefault(
            rowFieldOrDefault(row, "pciseq_enabled", params.pciseq_enabled),
            false,
            "pciseq_enabled for ${pairId}",
        ),
        ateraRow,
        row,
        "pciseq_enabled",
        "pciSeq",
        settingNotes,
    )
    def pciseqReference = pciseqReferenceSettings(row, species, params)
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
    def viewerCacheEnabled = boolOrDefault(
        rowFieldOrDefault(
            row,
            "viewer_cache_enabled",
            params.viewer_cache_enabled,
        ),
        true,
        "viewer_cache_enabled for ${pairId}",
    )
    def registeredImagesCsv = optionalNormalizedPathString(
        chooseField(row, ["xenium_registered_images_csv"])
    )
    def registerImagesEnabled = (
        activePlatforms.contains("XENIUM") && registeredImagesCsv != null
    )
    def registerPostXeniumIfNonrigid = boolOrDefault(
        rowFieldOrDefault(
            row,
            "register_post_xenium_if_nonrigid",
            params.register_post_xenium_if_nonrigid,
        ),
        false,
        "register_post_xenium_if_nonrigid for ${pairId}",
    )
    def spatialGeneAnalysisEnabled = boolOrDefault(
        rowFieldOrDefault(
            row,
            "spatial_gene_analysis_enabled",
            params.spatial_gene_analysis_enabled,
        ),
        true,
        "spatial_gene_analysis_enabled for ${pairId}",
    )
    def spatialGeneTranscriptAnalysisEnabled = boolOrDefault(
        rowFieldOrDefault(
            row,
            "spatial_gene_analysis_transcript_analysis_enabled",
            params.spatial_gene_analysis_transcript_analysis_enabled,
        ),
        true,
        "spatial_gene_analysis_transcript_analysis_enabled for ${pairId}",
    )
    def corticalDepthEnabled = ateraIgnoresGlobalSwitch(
        boolOrDefault(
            rowFieldOrDefault(
                row,
                "cortical_depth_enabled",
                params.cortical_depth_enabled,
            ),
            false,
            "cortical_depth_enabled for ${pairId}",
        ),
        ateraRow,
        row,
        "cortical_depth_enabled",
        "cortical depth",
        settingNotes,
    )
    def distanceFromObjectEnabled = ateraIgnoresGlobalSwitch(
        boolOrDefault(
            rowFieldOrDefault(
                row,
                "distance_from_object_enabled",
                params.distance_from_object_enabled,
            ),
            false,
            "distance_from_object_enabled for ${pairId}",
        ),
        ateraRow,
        row,
        "distance_from_object_enabled",
        "distance from object",
        settingNotes,
    )
    def mecrEnabled = boolOrDefault(
        rowFieldOrDefault(row, "mecr_enabled", params.mecr_enabled),
        species == "human",
        "mecr_enabled for ${pairId}",
    )
    def menderEnabled = boolOrDefault(
        rowFieldOrDefault(row, "mender_enabled", params.mender_enabled),
        false,
        "mender_enabled for ${pairId}",
    )
    def rowMenderSegmentations = chooseField(row, ["mender_segmentations"])
    def menderSegmentations = resolveRowSegmentations(
        rowMenderSegmentations,
        rowFieldOrDefault(row, "mender_segmentations", params.mender_segmentations),
        { value -> normalizeMenderSegmentations(value, segmentationSource) },
        segmentationSource,
        "mender_segmentations",
        pairId,
        settingNotes,
        activePlatforms,
    )
    def gastonEnabled = ateraIgnoresGlobalSwitch(
        boolOrDefault(
            rowFieldOrDefault(row, "gaston_enabled", params.gaston_enabled),
            false,
            "gaston_enabled for ${pairId}",
        ),
        ateraRow,
        row,
        "gaston_enabled",
        "GASTON",
        settingNotes,
    )
    def gastonSegmentations = normalizeGastonSegmentations(
        rowFieldOrDefault(
            row,
            "gaston_segmentations",
            params.gaston_segmentations,
        )
    )
    def gastonDomainMode = rowFieldOrDefault(
        row,
        "gaston_domain_mode",
        params.gaston_domain_mode,
    ).toString().trim().toLowerCase()
    if (!(gastonDomainMode in ["auto", "fixed"])) {
        throw new IllegalArgumentException(
            "Unknown gaston_domain_mode for ${pairId}: '${gastonDomainMode}'. " +
            "Use auto or fixed."
        )
    }
    def gastonKeepCheckpoints = rowFieldOrDefault(
        row,
        "gaston_keep_checkpoints",
        params.gaston_keep_checkpoints,
    ).toString().trim().toLowerCase()
    if (!(gastonKeepCheckpoints in ["none", "best", "all"])) {
        throw new IllegalArgumentException(
            "Unknown gaston_keep_checkpoints for ${pairId}: " +
            "'${gastonKeepCheckpoints}'. Use none, best, or all."
        )
    }
    def gastonSettings = [
        use_gpu: boolOrDefault(
            rowFieldOrDefault(row, "gaston_use_gpu", params.gaston_use_gpu),
            params.gaston_use_gpu,
            "gaston_use_gpu for ${pairId}",
        ),
        n_restarts: intOrDefault(
            rowFieldOrDefault(row, "gaston_n_restarts", params.gaston_n_restarts),
            params.gaston_n_restarts,
        ),
        epochs: intOrDefault(
            rowFieldOrDefault(row, "gaston_epochs", params.gaston_epochs),
            params.gaston_epochs,
        ),
        checkpoint_interval: intOrDefault(
            rowFieldOrDefault(
                row,
                "gaston_checkpoint_interval",
                params.gaston_checkpoint_interval,
            ),
            params.gaston_checkpoint_interval,
        ),
        hidden_spatial: positiveIntegerListOrDefault(
            rowFieldOrDefault(
                row,
                "gaston_hidden_spatial",
                params.gaston_hidden_spatial,
            ),
            [20, 20],
            "gaston_hidden_spatial for ${pairId}",
        ),
        hidden_expression: positiveIntegerListOrDefault(
            rowFieldOrDefault(
                row,
                "gaston_hidden_expression",
                params.gaston_hidden_expression,
            ),
            [20, 20],
            "gaston_hidden_expression for ${pairId}",
        ),
        optimizer: rowFieldOrDefault(
            row,
            "gaston_optimizer",
            params.gaston_optimizer,
        ).toString().trim().toLowerCase(),
        glmpca_dimensions: intOrDefault(
            rowFieldOrDefault(
                row,
                "gaston_glmpca_dimensions",
                params.gaston_glmpca_dimensions,
            ),
            params.gaston_glmpca_dimensions,
        ),
        glmpca_penalty: floatOrDefault(
            rowFieldOrDefault(
                row,
                "gaston_glmpca_penalty",
                params.gaston_glmpca_penalty,
            ),
            params.gaston_glmpca_penalty,
        ),
        glmpca_iterations: intOrDefault(
            rowFieldOrDefault(
                row,
                "gaston_glmpca_iterations",
                params.gaston_glmpca_iterations,
            ),
            params.gaston_glmpca_iterations,
        ),
        domain_mode: gastonDomainMode,
        num_domains: nullableIntOrDefault(
            chooseField(row, ["gaston_num_domains"]),
            params.gaston_num_domains,
        ),
        min_domains: intOrDefault(
            rowFieldOrDefault(
                row,
                "gaston_min_domains",
                params.gaston_min_domains,
            ),
            params.gaston_min_domains,
        ),
        max_domains: intOrDefault(
            rowFieldOrDefault(
                row,
                "gaston_max_domains",
                params.gaston_max_domains,
            ),
            params.gaston_max_domains,
        ),
        domain_buckets: intOrDefault(
            rowFieldOrDefault(
                row,
                "gaston_domain_buckets",
                params.gaston_domain_buckets,
            ),
            params.gaston_domain_buckets,
        ),
        auto_k_fallback: nullableIntOrDefault(
            chooseField(row, ["gaston_auto_k_fallback"]),
            params.gaston_auto_k_fallback,
        ),
        write_spatialdata_table: boolOrDefault(
            rowFieldOrDefault(
                row,
                "gaston_write_spatialdata_table",
                params.gaston_write_spatialdata_table,
            ),
            params.gaston_write_spatialdata_table,
            "gaston_write_spatialdata_table for ${pairId}",
        ),
        keep_seed_models: boolOrDefault(
            rowFieldOrDefault(
                row,
                "gaston_keep_seed_models",
                params.gaston_keep_seed_models,
            ),
            params.gaston_keep_seed_models,
            "gaston_keep_seed_models for ${pairId}",
        ),
        keep_checkpoints: gastonKeepCheckpoints,
        max_genes: intOrDefault(
            rowFieldOrDefault(row, "gaston_max_genes", params.gaston_max_genes),
            params.gaston_max_genes,
        ),
        figure_dpi: intOrDefault(
            rowFieldOrDefault(
                row,
                "gaston_figure_dpi",
                params.gaston_figure_dpi,
            ),
            params.gaston_figure_dpi,
        ),
    ]
    if (!(gastonSettings.optimizer in ["adam", "sgd", "adagrad"])) {
        throw new IllegalArgumentException(
            "Unknown gaston_optimizer for ${pairId}: " +
            "'${gastonSettings.optimizer}'. Use adam, sgd, or adagrad."
        )
    }
    if (gastonDomainMode == "fixed" && gastonSettings.num_domains == null) {
        throw new IllegalArgumentException(
            "gaston_num_domains is required for ${pairId} when " +
            "gaston_domain_mode=fixed"
        )
    }
    def distanceFromObjectSegmentations = normalizeDistanceFromObjectSegmentations(
        rowFieldOrDefault(
            row,
            "distance_from_object_segmentations",
            params.distance_from_object_segmentations,
        )
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
        spatialGeneAnalysisEnabled,
        corticalDepthEnabled,
        distanceFromObjectEnabled,
        viewerCacheEnabled,
        mecrEnabled,
        menderEnabled,
        registerImagesEnabled,
        gastonEnabled,
        pciseqEnabled,
        segmentationSource,
        activePlatforms,
        CellposeHitlSettings.enabled(params),
    )
    def startStage = normalizeStage(startStageRaw, startParamName)
    def stopStage = normalizeStage(stopStageRaw, stopParamName)
    validateStage(
        startStage,
        stageOrder,
        startParamName,
        alignmentEnabled,
        spatialGeneAnalysisEnabled,
        menderEnabled,
        segmentationSource,
        ateraRow,
    )
    validateStage(
        stopStage,
        stageOrder,
        stopParamName,
        alignmentEnabled,
        spatialGeneAnalysisEnabled,
        menderEnabled,
        segmentationSource,
        ateraRow,
    )
    // Opt-in terminal analyses extend the historical clustering stop default.
    // Explicit only_stage selections and non-default stop stages remain exact.
    def autoExtendedToMender = false
    def autoExtendedToGaston = false
    if (!onlyStageRaw && stopStage == "clustering_squidpy") {
        if (gastonEnabled && stageOrder.contains("gaston")) {
            stopStage = "gaston"
            autoExtendedToGaston = true
        } else if (menderEnabled && stageOrder.contains("mender")) {
            stopStage = "mender"
            autoExtendedToMender = true
        } else if (distanceFromObjectEnabled && stageOrder.contains("distance_from_object")) {
            stopStage = "distance_from_object"
        } else if (corticalDepthEnabled && stageOrder.contains("compute_cortical_depth")) {
            stopStage = "compute_cortical_depth"
        } else if (pciseqEnabled) {
            stopStage = "pciseq"
        }
    }
    if (stageOrder.indexOf(startStage) > stageOrder.indexOf(stopStage)) {
        error(
            "Samplesheet row ${pairId} has start_stage '${startStage}' " +
            "after stop_stage '${stopStage}'."
        )
    }

    def runBuild = stageInRange("build_spatialdata", startStage, stopStage, stageOrder)
    def runSegmentNuclei = stageInRange(
        "segment_nuclei",
        startStage,
        stopStage,
        stageOrder,
    )
    def runSegment = stageInRange("segment", startStage, stopStage, stageOrder)
    def runEnrich = stageInRange("enrich", startStage, stopStage, stageOrder)
    def runRegisterImages = stageInRange(
        "register_images",
        startStage,
        stopStage,
        stageOrder,
    )
    def runBuildViewerCaches = stageInRange(
        "build_viewer_caches",
        startStage,
        stopStage,
        stageOrder,
    )
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
    def runDistanceFromObject = stageInRange(
        "distance_from_object",
        startStage,
        stopStage,
        stageOrder,
    )
    def runQc = stageInRange("qc", startStage, stopStage, stageOrder)
    def runMecr = stageInRange("mecr", startStage, stopStage, stageOrder)
    def runAlign = stageInRange("align", startStage, stopStage, stageOrder)
    def runAlignQc = stageInRange("align_qc", startStage, stopStage, stageOrder)
    def runCompare = stageInRange("compare", startStage, stopStage, stageOrder)
    def runVisualize = stageInRange("visualize", startStage, stopStage, stageOrder)
    def runSpatialGeneAnalysis = stageInRange(
        "spatial_gene_analysis",
        startStage,
        stopStage,
        stageOrder,
    )
    def runClusteringSquidpy = stageInRange(
        "clustering_squidpy",
        startStage,
        stopStage,
        stageOrder,
    )
    def runMapMyCells = stageInRange("mapmycells", startStage, stopStage, stageOrder)
    def runPciseq = stageInRange("pciseq", startStage, stopStage, stageOrder)
    // Extending the historical clustering stop for MENDER must not implicitly
    // activate MapMyCells, which otherwise lies between clustering and MENDER.
    if (autoExtendedToMender || autoExtendedToGaston) {
        runMapMyCells = false
    }
    // ATERA rows have no standalone MapMyCells; an explicit stop at MENDER
    // passes over it like the extended clustering stop above. A row that names
    // mapmycells itself as its start or stop stage still selects it, and the
    // ATERA preflight refuses it.
    def ateraSkipsMapMyCells = (
        ateraRow && startStage != "mapmycells" && stopStage != "mapmycells"
    )
    if (ateraSkipsMapMyCells) {
        runMapMyCells = false
    }
    def runGaston = stageInRange("gaston", startStage, stopStage, stageOrder)
    def runMender = stageInRange("mender", startStage, stopStage, stageOrder)
    def requiredClusteringSegmentations = (
        analysisSegmentations + (runMender ? menderSegmentations : []) +
        (runGaston ? gastonSegmentations : [])
    ).unique()
    def analysisInputSegmentations = runClusteringSquidpy
        ? requiredClusteringSegmentations
        : analysisSegmentations
    if (menderEnabled) {
        validateMenderParams(params)
    }
    if (
        runMender &&
        runClusteringSquidpy &&
        !effectiveClusteringHierarchicalEnabled(params)
    ) {
        throw new IllegalArgumentException(
            "MENDER clustering prerequisites require " +
            "--clustering_squidpy_hierarchical_enabled true"
        )
    }
    def needAnalysisZarrs =
        runQc ||
        runAlign ||
        runAlignQc ||
        runMecr ||
        runCompare ||
        runVisualize ||
        runSpatialGeneAnalysis ||
        runClusteringSquidpy
    def needAlignmentDownstream =
        pairedMode &&
        alignmentEnabled &&
        (
            runCompare ||
            runVisualize ||
            runSpatialGeneAnalysis ||
            runClusteringSquidpy
        )
    def needAlignmentResults =
        pairedMode &&
        alignmentEnabled &&
        (runAlign || runAlignQc || needAlignmentDownstream)

    // rca-hook:H3: annotation settings and the map_first MAPMYCELLS rule (plan §3.1).
    // Legacy rows get no key, so their settings and -resume hashes are unchanged.
    def annotationSettings = AnnotationSettings.forRow(
        annotationModeRow(row, params), params, species
    )
    if (activePlatforms.contains("ATERA") && AnnotationSettings.isMapFirst(annotationSettings)) {
        annotationSettings.clustering_squidpy_table_key_suffix = "rcahippocampal"
        annotationSettings.mender_unassigned_state_policy = "exclude_from_features"
    }
    runMapMyCells = AnnotationSettings.runMapMyCells(runMapMyCells, annotationSettings, stopStage)
    return annotationSettings + [
        pair_id: pairId,
        species: species,
        analysis_mode: analysisMode,
        enable_alignment: alignmentEnabled,
        active_platforms: activePlatforms,
        segmentation_source: segmentationSource,
        setting_notes: settingNotes,
        paired_mode: pairedMode,
        analysis_segmentations: analysisSegmentations,
        analysis_input_segmentations: analysisInputSegmentations,
        required_clustering_segmentations: requiredClusteringSegmentations,
        mender_segmentations: menderSegmentations,
        gaston_segmentations: gastonSegmentations,
        gaston: gastonSettings,
        distance_from_object_segmentations: distanceFromObjectSegmentations,
        alignment_annotation_paths: [
            MERSCOPE: optionalNormalizedPathString(
                alignmentTissueAnnotationPath(row, "MERSCOPE")
            ),
            XENIUM: optionalNormalizedPathString(
                alignmentTissueAnnotationPath(row, "XENIUM")
            ),
        ],
        spatial_gene_analysis_enabled: spatialGeneAnalysisEnabled,
        spatial_gene_analysis_transcript_analysis_enabled: (
            spatialGeneAnalysisEnabled && spatialGeneTranscriptAnalysisEnabled
        ),
        stage_order: stageOrder,
        start_stage: startStage,
        stop_stage: stopStage,
        selected_stages: stageOrder.findAll { stage ->
            stageInRange(stage, startStage, stopStage, stageOrder) &&
                !(
                    (autoExtendedToMender || autoExtendedToGaston || ateraSkipsMapMyCells) &&
                    stage == "mapmycells"
                )
        },
        run_build: runBuild,
        run_segment_nuclei: runSegmentNuclei,
        run_segment: runSegment,
        run_native_segment: runSegment && nativeSegmentation,
        run_atera_cellpose: runSegment && runsAteraCellpose(segmentationSource, activePlatforms),
        run_atera_proseg: runSegment && runsAteraProseg(segmentationSource, activePlatforms),
        run_enrich: runEnrich,
        run_register_images: runRegisterImages,
        registered_images_csv: registeredImagesCsv,
        register_post_xenium_if_nonrigid: registerPostXeniumIfNonrigid,
        run_build_viewer_caches: runBuildViewerCaches,
        run_mask_image_quantification: runMaskImageQuantification,
        run_compute_cortical_depth: runComputeCorticalDepth,
        run_distance_from_object: runDistanceFromObject,
        run_qc: runQc,
        run_mecr: runMecr,
        run_align: runAlign,
        run_align_qc: runAlignQc,
        run_compare: runCompare,
        run_visualize: runVisualize,
        run_spatial_gene_analysis: runSpatialGeneAnalysis,
        run_clustering_squidpy: runClusteringSquidpy,
        run_pciseq: runPciseq,
        pciseq_reference_path: pciseqReference.reference_path,
        pciseq_reference: pciseqReference,
        pciseq_reference_key: pciseqReferenceKey(pciseqReference),
        run_mender: runMender,
        run_mapmycells: runMapMyCells,
        mender_published_output_mode: runMender && !runClusteringSquidpy,
        mender_bypass_terminal_gate: (
            runMender &&
            (
                (onlyStageRaw && startStage == "mender") ||
                (!runMapMyCells &&
                 !runDistanceFromObject &&
                 !runComputeCorticalDepth &&
                 !runClusteringSquidpy)
            )
        ),
        run_terminal_analyses: runMender || runGaston,
        run_gaston: runGaston,
        gaston_published_output_mode: runGaston && !runClusteringSquidpy,
        gaston_bypass_terminal_gate: (
            runGaston &&
            (
                (onlyStageRaw && startStage == "gaston") ||
                (!runMapMyCells &&
                 !runDistanceFromObject &&
                 !runComputeCorticalDepth &&
                 !runClusteringSquidpy)
            )
        ),
        need_build_results: runSegmentNuclei || runSegment || runEnrich,
        need_enriched_zarrs: (
            runRegisterImages ||
            runBuildViewerCaches ||
            runMaskImageQuantification ||
            runComputeCorticalDepth ||
            runDistanceFromObject ||
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
    def pipelineSpecies = normalizeSpecies(params.species)
    def mapMyCellsReferenceMode = params.mapmycells_reference_mode == null
        ? (pipelineSpecies == "mouse" ? "whole_brain" : "both")
        : params.mapmycells_reference_mode.toString().trim().toLowerCase()
    def mapMyCellsPlotsOnly = params.mapmycells_plots_only == null
        ? false
        : params.mapmycells_plots_only.toString().trim().toLowerCase() == "true"
    def mapMyCellsReferenceAtlas = params.mapmycells_reference_atlas == null
        ? (pipelineSpecies == "mouse" ? "wmb" : "whb")
        : params.mapmycells_reference_atlas.toString().trim().toLowerCase()
    def mapMyCellsQuerySpecies = params.mapmycells_query_species == null
        ? pipelineSpecies
        : params.mapmycells_query_species.toString().trim().toLowerCase()
    def mapMyCellsAutoDownload = params.mapmycells_auto_download_references == null
        ? true
        : params.mapmycells_auto_download_references.toString().trim().toLowerCase() == "true"
    def markerLookupPath = params.mapmycells_marker_lookup_path ?:
        (mapMyCellsReferenceAtlas == "whb" ? params.mapmycells_whb_marker_lookup_path : null)
    def precomputedStatsPath = params.mapmycells_precomputed_stats_path ?:
        (mapMyCellsReferenceAtlas == "whb" ? params.mapmycells_whb_precomputed_stats_path : null)
    if (!(mapMyCellsReferenceMode in ["whole_brain", "region", "both"])) {
        throw new IllegalArgumentException(
            "Invalid MAPMYCELLS --mapmycells_reference_mode " +
            "'${params.mapmycells_reference_mode}'. Valid values: " +
            "whole_brain, region, both"
        )
    }
    if (!(mapMyCellsReferenceAtlas in ["whb", "wmb"])) {
        throw new IllegalArgumentException(
            "Invalid MAPMYCELLS --mapmycells_reference_atlas " +
            "'${params.mapmycells_reference_atlas}'. Valid values: whb, wmb"
        )
    }
    if (!(mapMyCellsQuerySpecies in ["human", "mouse"])) {
        throw new IllegalArgumentException(
            "Invalid MAPMYCELLS --mapmycells_query_species " +
            "'${params.mapmycells_query_species}'. Valid values: human, mouse"
        )
    }
    if (mapMyCellsQuerySpecies != pipelineSpecies) {
        throw new IllegalArgumentException(
            "MAPMYCELLS query species '${mapMyCellsQuerySpecies}' does not match " +
            "pipeline --species ${pipelineSpecies}"
        )
    }
    if (pipelineSpecies == "mouse" && mapMyCellsReferenceAtlas != "wmb") {
        throw new IllegalArgumentException(
            "Mouse MAPMYCELLS queries currently require " +
            "--mapmycells_reference_atlas wmb"
        )
    }
    if (!mapMyCellsPlotsOnly &&
        mapMyCellsReferenceMode in ["whole_brain", "both"] &&
        !mapMyCellsAutoDownload &&
        !markerLookupPath) {
        throw new IllegalArgumentException(
            "Missing required parameter for MAPMYCELLS: --mapmycells_marker_lookup_path"
        )
    }
    if (!mapMyCellsPlotsOnly &&
        mapMyCellsReferenceMode in ["whole_brain", "both"] &&
        !mapMyCellsAutoDownload &&
        !precomputedStatsPath) {
        throw new IllegalArgumentException(
            "Missing required parameter for MAPMYCELLS: --mapmycells_precomputed_stats_path"
        )
    }
    if (!mapMyCellsPlotsOnly &&
        mapMyCellsReferenceAtlas == "wmb" &&
        mapMyCellsQuerySpecies == "human" &&
        !mapMyCellsAutoDownload &&
        !params.mapmycells_gene_mapping_db_path) {
        throw new IllegalArgumentException(
            "Human-to-WMB MAPMYCELLS requires " +
            "--mapmycells_gene_mapping_db_path when automatic downloads are disabled"
        )
    }
    if (!mapMyCellsPlotsOnly &&
        mapMyCellsReferenceMode in ["region", "both"] &&
        pipelineSpecies == "mouse" &&
        !params.mapmycells_region_labels) {
        throw new IllegalArgumentException(
            "Missing required parameter for MAPMYCELLS region mode: " +
            "--mapmycells_region_labels"
        )
    }
}

def validateMenderParams(params) {
    def mode = params.mender_clustering_mode.toString().trim().toLowerCase()
    if (!(mode in ["resolution", "target_k"])) {
        throw new IllegalArgumentException(
            "Invalid --mender_clustering_mode '${params.mender_clustering_mode}'. " +
            "Use resolution or target_k."
        )
    }
    if ((params.mender_radius_um as float) <= 0.0) {
        throw new IllegalArgumentException("--mender_radius_um must be positive")
    }
    if ((params.mender_n_scales as int) <= 0) {
        throw new IllegalArgumentException("--mender_n_scales must be positive")
    }
    if ((params.mender_leiden_resolution as float) <= 0.0) {
        throw new IllegalArgumentException(
            "--mender_leiden_resolution must be positive"
        )
    }
    if (mode == "target_k" &&
        (params.mender_target_k == null || (params.mender_target_k as int) < 2)) {
        throw new IllegalArgumentException(
            "--mender_target_k must be at least 2 in target_k mode"
        )
    }
    if (!(params.mender_nn_mode.toString() in ["radius", "k", "ring"])) {
        throw new IllegalArgumentException("Invalid --mender_nn_mode")
    }
    if (!(params.mender_count_rep.toString() in ["s", "a"])) {
        throw new IllegalArgumentException("Invalid --mender_count_rep")
    }
    if (params.mender_missing_state_policy.toString() != "error") {
        throw new IllegalArgumentException(
            "Invalid --mender_missing_state_policy; only strict 'error' is supported"
        )
    }
    if (!params.mender_cell_state_key?.toString()?.trim()) {
        throw new IllegalArgumentException("--mender_cell_state_key must not be empty")
    }
}

// The SegmentationConfig of one row and platform (CELLPOSE_SEGMENT,
// CELLPOSE_NUCLEI_SEGMENT, PROSEG_SEGMENT and, for ATERA, ATERA_CELLPOSE_SEGMENT).
// ATERA (AT-7) segments the native morphology image of its built store on its
// Cellpose channels (DAPI + 18S): no transform or spec path (the pixel size
// comes from the store's spec sidecars), no ProSeg transcript CSV (the tiled
// ProSeg route reads the store's parquet) and no Cellpose nuclei. The XENIUM
// and MERSCOPE maps are built as before, key for key.
def segmentationConfigMap(pairId, platform, sourceSpatialdata, meta, params) {
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
    def persistentCellprobPath = file(
        publishedDatasetPath(
            params.outdir,
            pairId,
            platform,
            "segmentation/cellpose_cellprobs_tiled.npy",
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
    def persistentNucleiMaskPath = file(
        publishedDatasetPath(
            params.outdir,
            pairId,
            platform,
            "segmentation/cellpose_nuclei_masks_tiled.npy",
        )
    ).toAbsolutePath().toString()
    def persistentNucleiStitchingStatsPath = file(
        publishedDatasetPath(
            params.outdir,
            pairId,
            platform,
            "segmentation/cellpose_nuclei_stitching_stats.json",
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
            use_bfloat16: params.cellpose_use_bfloat16,
        ],
        nuclei_cellpose: [
            model_type: "nuclei",
            gpu: params.cellpose_gpu,
            diameter: params.cellpose_diameter,
            flow_threshold: params.cellpose_flow_threshold,
            cellprob_threshold: params.cellpose_cellprob,
            tile_overlap: params.cellpose_tile_overlap,
            bsize: params.cellpose_bsize,
            factor_rescale: 1.0,
            use_bfloat16: params.cellpose_use_bfloat16,
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
            geometry_filter: CellposeSettings.geometryFilter(params),
        ],
        nuclei_mask_filter: [
            final_min_area_um2: params.cellpose_final_min_area_um2,
            final_max_area_um2: params.cellpose_final_max_area_um2,
            final_filter_chunk_mb: params.cellpose_final_filter_chunk_mb,
            geometry_filter: CellposeSettings.geometryFilter(params),
        ],
        proseg: [
            binary_path: params.proseg_binary,
            samples: params.proseg_samples,
            voxel_size: params.proseg_voxel_size,
            burnin_voxel_size: params.proseg_burnin_voxel_size,
            nuclear_reassignment_prob: params.proseg_nuclear_reassignment_prob,
            diffusion_probability: params.proseg_diffusion_probability,
            cell_compactness: params.proseg_cell_compactness,
            num_threads: params.proseg_num_threads,
            voxel_layers: 2,
        ],
        proseg_hybrid: [
            enabled: params.proseg_hybrid_enabled,
            min_transcripts: params.proseg_hybrid_min_transcripts,
            outlier_neighbors: params.proseg_hybrid_outlier_neighbors,
            outlier_mad_multiplier: params.proseg_hybrid_outlier_mad_multiplier,
            minimum_external_group: params.proseg_hybrid_minimum_external_group,
            chain_radius_scale: params.proseg_hybrid_chain_radius_scale,
            near_surface_radius_fraction: params.proseg_hybrid_near_surface_radius_fraction,
            maximum_expansion_radius_fraction: params.proseg_hybrid_maximum_expansion_radius_fraction,
            attachment_arc_width_scale: params.proseg_hybrid_attachment_arc_width_scale,
            rounding_radius_fraction: params.proseg_hybrid_rounding_radius_fraction,
            smoothing_radius_um: params.proseg_hybrid_smoothing_radius_um,
            outward_rounding_um: params.proseg_hybrid_outward_rounding_um,
            smoothing_quad_segs: params.proseg_hybrid_smoothing_quad_segs,
            containment_tolerance_um: params.proseg_hybrid_containment_tolerance_um,
        ],
        memory: [
            max_system_ram_gb: params.max_ram_gb,
            memory_warn_gb: params.warn_ram_gb,
            transcript_chunk_rows: params.transcript_chunk_rows,
        ],
    ]

    // hitl-hook:C4
    if (meta.cellpose_settings) {
        baseConfig.cellpose.putAll(meta.cellpose_settings.cellpose)
        baseConfig.mask_filter.putAll(meta.cellpose_settings.mask_filter)
        if (meta.cellpose_settings.tiling) {
            baseConfig.tiling.putAll(meta.cellpose_settings.tiling)
        }
        if (meta.cellpose_settings.cellpose_model_policy) {
            baseConfig.cellpose_model_policy = meta.cellpose_settings.cellpose_model_policy
        }
    }

    // Only present for rows with bead removal on (see segmentMetaForPlatform).
    if (meta.bead_mask_veto) {
        baseConfig.bead_mask_veto = meta.bead_mask_veto
    }

    def segmentConfig = baseConfig + [
        dataset: [
            name: "${pairId}_${platform}",
            platform: platform,
            data_path: sourceSpatialdata.toString(),
            channels: meta.channels,
            output_dir: "segment_out",
            persistent_latest_zarr_path: persistentLatestZarrPath,
            persistent_mask_path: persistentMaskPath,
            persistent_cellpose_cellprob_path: persistentCellprobPath,
            persistent_transcripts_path: persistentTranscriptsPath,
            persistent_cellpose_stitching_stats_path: persistentStitchingStatsPath,
            persistent_nuclei_mask_path: persistentNucleiMaskPath,
            persistent_nuclei_stitching_stats_path: persistentNucleiStitchingStatsPath,
            image_prefix: meta.image_prefix,
            z_range: meta.z_range,
            transform_path: meta.transform_path,
            xenium_spec_path: meta.xenium_spec_path,
            min_qv: meta.min_qv,
            proseg_overrides: [voxel_layers: meta.voxel_layers],
        ],
    ]
    if (platform == "ATERA") {
        def dataset = segmentConfig.dataset
        ["persistent_transcripts_path", "persistent_nuclei_mask_path",
            "persistent_nuclei_stitching_stats_path", "image_prefix", "z_range",
            "transform_path", "xenium_spec_path", "min_qv"].each { name ->
            dataset.remove(name)
        }
        dataset.image_key = meta.image_key
        dataset.proseg_transcripts = "none"
        if (AteraDensitySettings.enabledForMeta(meta, params)) {
            segmentConfig.cellpose.input_normalization = "per_channel"
            dataset.transcript_density_path = meta.transcript_density_path ?: publishedDatasetPath(
                params.outdir, pairId, platform, "transcript_density/transcript_density.zarr")
        }
        segmentConfig.remove("nuclei_cellpose")
        segmentConfig.remove("nuclei_mask_filter")
        segmentConfig.remove("proseg_hybrid")
    }
    return segmentConfig
}

workflow {
    // Private site configs (docs/local-configuration.md): stop a launch whose
    // site locations are missing instead of letting null params fall back.
    def siteConfigProblems = SiteConfigCheck.problems(
        workflow.profile,
        workflow.containerEngine,
        params,
        System.getenv(),
    )
    if (siteConfigProblems) {
        error(siteConfigProblems.join("\n"))
    }

    if (!params.samplesheet) {
        error "Missing required parameter: --samplesheet"
    }

    samplesheet_ch = channel
        .fromPath(params.samplesheet, checkIfExists: true)
        .splitCsv(header: true, sep: ",", quote: '"', strip: true)

    // rca-site:H5 (pre-registration §20 D15): each row's clustering_squidpy_mode
    // column, read once here because the clustering wiring is chosen before
    // any channel runs. A run takes the map_first wiring when any row
    // resolves to map_first; a run whose rows are all legacy keeps the legacy
    // DAG.
    def samplesheetRows = file(params.samplesheet, checkIfExists: true)
        .splitCsv(header: true, sep: ",", quote: '"', strip: true)
    // ATERA rows without their own mode column cluster in legacy mode
    // (annotationModeRow).
    def annotationModeRows = samplesheetRows.collect { row -> annotationModeRow(row, params) }
    AnnotationSettings.useRowModes(
        annotationModeRows.collect { row -> row[AnnotationDefaults.ROW_MODE_COLUMN] }
    )
    // The mode summary and the legacy deprecation warning, which leaves out
    // every ATERA row (annotationModeLaunchLog).
    def annotationModeLog = annotationModeLaunchLog(samplesheetRows, params)
    annotationModeLog.info.each { line -> log.info(line) }
    def nearMissModeColumns = AnnotationSettings.nearMissModeColumns(
        samplesheetRows ? samplesheetRows[0].keySet() : []
    )
    if (nearMissModeColumns) {
        log.warn(
            "Samplesheet column(s) ${nearMissModeColumns.join(', ')} are ignored: the " +
            "per-row mode column is '${AnnotationDefaults.ROW_MODE_COLUMN}' (exact name). " +
            "Those rows follow the run's clustering mode."
        )
    }
    // Legacy rows of a flipped species (human since M8, mouse since M9); no
    // ATERA row is a deprecated choice.
    annotationModeLog.warn.each { line -> log.warn(line) }

    sample_rows_raw_ch = samplesheet_ch.map { row ->
        def settings = rowSampleSettings(row, params)
        log.info(
            "Sample ${settings.pair_id}: analysis_mode=${settings.analysis_mode}; " +
            "segmentation_source=${settings.segmentation_source}; " +
            "enable_alignment=${settings.enable_alignment}; " +
            "spatial_gene_analysis=${settings.run_spatial_gene_analysis}; " +
            "spatial_gene_transcripts=" +
            "${settings.spatial_gene_analysis_transcript_analysis_enabled}; " +
            "cortical_depth=${settings.run_compute_cortical_depth}; " +
            "distance_from_object=${settings.run_distance_from_object}; " +
            "active platforms=${settings.active_platforms.join(', ')}; " +
            "analysis segmentations=${settings.analysis_segmentations.join(', ')}; " +
            "distance segmentations=" +
            "${settings.distance_from_object_segmentations.join(', ')}; " +
            "mender=${settings.run_mender}; " +
            "pciseq=${settings.run_pciseq}; " +
            "pciseq reference=${settings.pciseq_reference.source}; " +
            "pciseq regions=${settings.pciseq_reference.region_labels.join(', ')}; " +
            "mender segmentations=${settings.mender_segmentations.join(', ')}; " +
            "required clustering segmentations=" +
            "${settings.required_clustering_segmentations.join(', ')}; " +
            "GASTON segmentations=${settings.gaston_segmentations.join(', ')}; " +
            "selected stages=${settings.selected_stages.join(' -> ')}"
        )
        settings.setting_notes.each { note ->
            log.info("Sample ${settings.pair_id}: ${note}")
        }
        tuple(settings.pair_id, row, settings)
    }

    // rca-hook:H10: --annotation_prepare_only builds reference bundles (plan §3.2)
    // and runs no pipeline stage; legacy and map_first runs skip this block.
    if (AnnotationReferences.prepareOnly(params)) {
        sample_rows_raw_ch = sample_rows_raw_ch.filter { entry -> !ateraPrepareOnlyRow(entry) }
        if (params.annotation_hippocampal_resources) {
            HIPPOCAMPAL_PREPARE_ONLY(
                channel.value(file(params.annotation_hippocampal_resources)),
                channel.value(AnnotationReferences.sourcesFingerprint("${projectDir}/../src", ["merxen/annotation/hippocampal_pipeline.py", "merxen/annotation/hippocampal_profile.py"])),
            )
        }
        ANNOTATION_PREPARE_ONLY(sample_rows_raw_ch)
        sample_rows_raw_ch = channel.empty()
    }

    preflight_done_ch = sample_rows_raw_ch
        .map { _pairId, row, settings ->
            runPreflightChecks(row, settings, params)
            ("MERSCOPE" in settings.active_platforms) &&
                merscopeBeadRemovalRequested(row, settings.pair_id, params)
        }
        .collect()
        .map { beadRequested ->
            // The fill is a global switch and only acts on samples that have
            // bead removal on, so say once that it is active, and only when
            // some sample asks for bead removal.
            if (
                beadRequested.any { requested -> requested } &&
                merscopeBeadRemovalFillRequested(params)
            ) {
                log.info(
                    "merscope_bead_removal_fill is on (mode " +
                    "${merscopeBeadRemovalFillMode(params)}): samples with bead " +
                    "removal also get the residual-footprint fill"
                )
            }
            true
        }

    sample_rows_ch = sample_rows_raw_ch
        .combine(preflight_done_ch)
        .map { pairId, row, settings, _doneFlag ->
            tuple(pairId, row, settings)
        }

    // Optional model training completes before any segmentation input is emitted.
    // Normal runs retain the existing row channel and task configuration text.
    if (CellposeHitlSettings.enabled(params)) {
        def hitlSpec = CellposeHitlSettings.specification(params)
        hitl_training_inputs_ch = sample_rows_raw_ch.collect(flat: false)
            .map { entries ->
                CellposeHitlSettings.validateRows(entries, hitlSpec, params)
                tuple(CellposeHitlSettings.configJson(hitlSpec),
                    hitlSpec.package_paths.collect { path -> file(path) })
            }
            .combine(preflight_done_ch)
            .map { configJson, packages, _done -> tuple(configJson, packages) }
        hitl_model_ready_ch = CELLPOSE_HITL_TRAIN(hitl_training_inputs_ch).model_selector
            .map { selectorFile -> new groovy.json.JsonSlurper().parse(selectorFile.toFile()) }
            .first()
        sample_rows_ch = sample_rows_ch.combine(hitl_model_ready_ch)
            .map { pairId, row, settings, selector ->
                tuple(pairId,
                    CellposeHitlSettings.applyTrainedModel(row, settings, selector, params),
                    settings)
            }
    }

    def hitlSeedRequested = params.cellpose_hitl_source_outdir || params.cellpose_hitl_source_zarr ||
        samplesheetRows.any { row -> row.any { key, value -> key.toString().endsWith('_hitl_source_zarr') && value?.toString()?.trim() } }
    if (hitlSeedRequested) {
        hitl_seed_inputs_ch = sample_rows_ch.flatMap { pairId, row, settings ->
            settings.active_platforms.collect { platform ->
                def seed = CellposeHitlSettings.seedConfig(pairId, platform, row, settings, params)
                seed ? tuple(groovy.json.JsonOutput.prettyPrint(groovy.json.JsonOutput.toJson(seed)),
                    file(seed.source_zarr), seed.nuclei_source_dir
                        ? new File(seed.nuclei_source_dir.toString()).listFiles()
                            .findAll { it.isFile() && it.name.startsWith('cellpose_nuclei') }
                            .collect { file(it.toString()) }
                        : []) : null
            }.findAll { it != null }
        }
        hitl_seed_complete_ch = CELLPOSE_HITL_SEED(hitl_seed_inputs_ch).complete.collect().map { true }
        sample_rows_ch = sample_rows_ch.combine(hitl_seed_complete_ch)
            .map { pairId, row, settings, _seeded -> tuple(pairId, row, settings) }
    }

    // rca-hook:H6: end-of-run annotation summary (plan §3.1); empty in legacy runs.
    def annotationCompletionParams = params
    def annotationCompletionWorkflow = workflow
    annotationCompletionWorkflow.onComplete {
        def annotationSummary = AnnotationSettings.completionSummary(
            annotationCompletionParams,
            AnnotationRunRecord.runInfo() + [success: annotationCompletionWorkflow.success],
        )
        if (annotationSummary) {
            log.info(annotationSummary)
        }
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
        // Native rows and every ATERA row run NATIVE_LATEST instead (an ATERA
        // cellpose_proseg row runs ATERA_CELLPOSE_SEGMENT and the tiled ProSeg
        // route below, never SEGMENT).
        if (
            !(settings.run_segment_nuclei || settings.run_segment) ||
            usesNativeLatest(settings.segmentation_source, settings.active_platforms)
        ) {
            []
        } else {
            settings.active_platforms.collect { platform ->
                tuple(
                    "${pairId}|${platform}",
                    segmentMetaForPlatform(row, platform, params),
                    settings.run_segment_nuclei,
                    settings.run_segment,
                )
            }
        }
    }

    // Native rows (plan AT-6): NATIVE_LATEST turns the vendor segmentation of
    // the source store into the latest store and emits the SEGMENT tuple shape
    // with empty placeholders in the Cellpose mask, transcript CSV and nuclei
    // mask slots. The gate carries the row's source: a native_proseg row's
    // latest store goes on to the ATERA ProSeg stitch (AT-9) instead.
    native_latest_gate_ch = sample_rows_ch.flatMap { pairId, _row, settings ->
        if (!settings.run_native_segment) {
            []
        } else {
            settings.active_platforms.collect { platform ->
                tuple("${pairId}|${platform}", settings.segmentation_source)
            }
        }
    }

    native_latest_inputs_ch = build_results_ch
        .join(native_latest_gate_ch)
        .map { key, pairId, platform, sourceSpatialdata, _rowSource ->
            tuple(
                key,
                pairId,
                platform,
                groovy.json.JsonOutput.prettyPrint(
                    groovy.json.JsonOutput.toJson(
                        nativeLatestConfigForPlatform(pairId, platform, params)
                    )
                ),
                sourceSpatialdata,
            )
        }

    native_task_results_ch = NATIVE_LATEST(native_latest_inputs_ch).segmentation

    native_routed_ch = native_task_results_ch
        .join(native_latest_gate_ch)
        .branch {
            key, pairId, platform, latestZarr, maskPath, transcriptsCsv,
            nucleiMaskPath, rowSource ->
                proseg: runsAteraProseg(rowSource, [platform])
                    return tuple(
                        key, pairId, platform, latestZarr, maskPath, transcriptsCsv,
                        nucleiMaskPath, rowSource,
                    )
                plain: true
                    return tuple(
                        key, pairId, platform, latestZarr, maskPath, transcriptsCsv,
                        nucleiMaskPath,
                    )
        }
    native_plain_ch = native_routed_ch.plain

    // ATERA Cellpose (segmentation_source=cellpose_proseg, plan AT-7): Cellpose
    // segments the native morphology image of the built store on DAPI + 18S, as
    // on XENIUM, under the GPU lock. It writes no ProSeg transcript CSV (the
    // tiled ProSeg route reads the store's parquet) and no Cellpose nuclei
    // (ATERA keeps the vendor nuclei).
    atera_cellpose_meta_ch = sample_rows_ch.flatMap { pairId, row, settings ->
        if (!settings.run_atera_cellpose) {
            []
        } else {
            settings.active_platforms.findAll { platform -> platform == "ATERA" }
                .collect { platform ->
                    tuple(
                        "${pairId}|${platform}",
                        segmentMetaForPlatform(row, platform, params),
                    )
                }
        }
    }

    atera_cellpose_sources_ch = build_results_ch
        .join(atera_cellpose_meta_ch)
    atera_density_inputs_ch = atera_cellpose_sources_ch
        .filter { _key, _pairId, _platform, _source, meta -> AteraDensitySettings.enabledForMeta(meta, params) }
        .map { key, pairId, platform, sourceSpatialdata, meta ->
            tuple(key, pairId, platform,
                groovy.json.JsonOutput.prettyPrint(groovy.json.JsonOutput.toJson(
                    AteraDensitySettings.config(sourceSpatialdata, params,
                        AteraDensitySettings.selectedProfile(meta.cellpose_settings ?: [:], params)))),
                sourceSpatialdata)
        }
    atera_density_results_ch = ATERA_TRANSCRIPT_DENSITY(atera_density_inputs_ch).density
    atera_density_ready_ch = atera_cellpose_sources_ch
        .join(atera_density_results_ch)
        .map { key, pairId, platform, sourceSpatialdata, meta, _densityPair, _densityPlatform, densityPath, _densityManifest ->
            tuple(key, pairId, platform, sourceSpatialdata,
                meta + [transcript_density_path: densityPath.toRealPath().toString()], densityPath)
        }
    atera_density_disabled_ch = atera_cellpose_sources_ch
        .filter { _key, _pairId, _platform, _source, meta -> !AteraDensitySettings.enabledForMeta(meta, params) }
        .map { key, pairId, platform, sourceSpatialdata, meta ->
            tuple(key, pairId, platform, sourceSpatialdata, meta, [])
        }
    atera_cellpose_inputs_ch = atera_density_ready_ch.mix(atera_density_disabled_ch)
        .map { key, pairId, platform, sourceSpatialdata, meta, densityPath ->
            tuple(
                key,
                pairId,
                platform,
                groovy.json.JsonOutput.prettyPrint(
                    groovy.json.JsonOutput.toJson(
                        segmentationConfigMap(
                            pairId, platform, sourceSpatialdata, meta, params,
                        )
                    )
                ),
                densityPath,
            )
        }

    atera_cellpose_ch = ATERA_CELLPOSE_SEGMENT(atera_cellpose_inputs_ch).segmentation

    // A controlled stock-weight comparison; only the main retrained masks feed ProSeg.
    atera_comparison_gate_ch = sample_rows_ch.flatMap { pairId, _row, settings ->
        settings.run_atera_cellpose && settings.segmentation_source == 'cellpose_comparison'
            ? [tuple("${pairId}|ATERA", true)] : []
    }
    atera_stock_cellpose_inputs_ch = atera_cellpose_inputs_ch
        .join(atera_comparison_gate_ch)
        .map { key, pairId, platform, configJson, densityPath, _comparison ->
            def config = CellposeSettings.stockComparisonConfig(
                new groovy.json.JsonSlurper().parseText(configJson))
            tuple(key, pairId, platform,
                groovy.json.JsonOutput.prettyPrint(groovy.json.JsonOutput.toJson(config)), densityPath)
        }
    atera_stock_cellpose_ch = ATERA_STOCK_CELLPOSE_SEGMENT(atera_stock_cellpose_inputs_ch).segmentation

    // ATERA ProSeg route (plan AT-9; tiled, 2 mm tiles): the plan reads the
    // source store and the prior: the vendor cell mask (native_proseg, layer
    // proseg_vendor) or the ATERA Cellpose mask and logits (cellpose_proseg,
    // AT-7, layer reseg). One ATERA_PROSEG_TILE task per tile runs ProSeg on
    // its window (bounded by atera_proseg_tile_max_forks); ATERA_PROSEG_STITCH
    // writes the stitched cells into the latest store NATIVE_LATEST built and
    // emits the SEGMENT tuple, so ENRICH and the later stages take it as they
    // take NATIVE_LATEST's output. The gate carries the prior kind.
    atera_proseg_gate_ch = sample_rows_ch.flatMap { pairId, _row, settings ->
        if (!settings.run_atera_proseg) {
            []
        } else {
            settings.active_platforms.collect { platform ->
                tuple(
                    "${pairId}|${platform}",
                    AteraProsegLayers.priorKindForSource(settings.segmentation_source),
                )
            }
        }
    }

    atera_proseg_plan_inputs_ch = build_results_ch
        .join(atera_proseg_gate_ch.filter { _key, priorKind -> priorKind == "vendor" })
        .map { key, pairId, platform, sourceSpatialdata, _priorKind ->
            tuple(
                key,
                pairId,
                platform,
                groovy.json.JsonOutput.prettyPrint(
                    groovy.json.JsonOutput.toJson(
                        ateraProsegPlanConfig(pairId, platform, params)
                    )
                ),
                sourceSpatialdata,
            )
        }

    // A Cellpose plan stages the Cellpose mask and logits as path inputs, so
    // a re-run Cellpose invalidates the plan (and, through the files' identity
    // in the tile configs, every tile).
    atera_proseg_cellpose_plan_inputs_ch = build_results_ch
        .join(atera_proseg_gate_ch.filter { _key, priorKind -> priorKind == "cellpose" })
        .join(atera_cellpose_ch)
        .map {
            key, pairId, platform, sourceSpatialdata, _priorKind, _segPairId,
            _segPlatform, cellposeMask, cellposeCellprob, _transforms, _stats ->
                tuple(
                    key,
                    pairId,
                    platform,
                    groovy.json.JsonOutput.prettyPrint(
                        groovy.json.JsonOutput.toJson(
                            ateraProsegPlanConfig(pairId, platform, params, "cellpose")
                        )
                    ),
                    sourceSpatialdata,
                    cellposeMask,
                    cellposeCellprob,
                )
        }

    atera_proseg_plan_ch = ATERA_PROSEG_PLAN(atera_proseg_plan_inputs_ch).plan
        .mix(ATERA_PROSEG_PLAN_CELLPOSE(atera_proseg_cellpose_plan_inputs_ch).plan)

    // One tile task per tile the plan runs. The tile config is a value holding
    // the real path of the source store, so a re-run plan with the same
    // content leaves finished tiles cached.
    atera_proseg_tile_specs_ch = atera_proseg_plan_ch
        .join(build_results_ch)
        .flatMap { key, pairId, platform, planFile, _buildPairId, _buildPlatform, sourceSpatialdata ->
            def sourcePath = java.nio.file.Paths
                .get(sourceSpatialdata.toString())
                .toRealPath()
                .toString()
            def specs = ateraProsegTileSpecs(
                planFile.text, pairId, platform, sourcePath, params,
            )
            if (!specs) {
                log.warn(
                    "${pairId}:${platform}: the ATERA ProSeg plan runs no tile; " +
                    "the ProSeg layer is not built"
                )
            }
            specs.collect { tileId, tileConfig ->
                tuple(
                    key,
                    pairId,
                    platform,
                    specs.size(),
                    tileId,
                    groovy.json.JsonOutput.prettyPrint(
                        groovy.json.JsonOutput.toJson(tileConfig)
                    ),
                )
            }
        }

    segmentation_config_inputs_ch = build_results_ch
        .join(segment_meta_ch)
        .map {
            key, pairId, platform, sourceSpatialdata, meta,
            runSegmentNuclei, runSegment ->
            def segmentConfig = segmentationConfigMap(
                pairId, platform, sourceSpatialdata, meta, params,
            )

            tuple(
                key,
                pairId,
                platform,
                groovy.json.JsonOutput.prettyPrint(groovy.json.JsonOutput.toJson(segmentConfig)),
                runSegmentNuclei,
                runSegment,
            )
        }

    nuclei_segment_inputs_ch = segmentation_config_inputs_ch
        .filter {
            _key, _pairId, _platform, _segmentConfigJson,
            runSegmentNuclei, _runSegment ->
                runSegmentNuclei
        }
        .map {
            key, pairId, platform, segmentConfigJson,
            _runSegmentNuclei, _runSegment ->
                tuple(key, pairId, platform, segmentConfigJson)
        }

    cellpose_segment_inputs_ch = segmentation_config_inputs_ch
        .filter {
            _key, _pairId, _platform, _segmentConfigJson,
            _runSegmentNuclei, runSegment ->
                runSegment
        }
        .map {
            key, pairId, platform, segmentConfigJson,
            _runSegmentNuclei, _runSegment ->
                tuple(key, pairId, platform, segmentConfigJson)
        }

    nuclei_task_results_ch = CELLPOSE_NUCLEI_SEGMENT(nuclei_segment_inputs_ch).segmentation

    nuclei_published_results_ch = segmentation_config_inputs_ch
        .filter {
            _key, _pairId, _platform, _segmentConfigJson,
            runSegmentNuclei, runSegment ->
                runSegment && !runSegmentNuclei
        }
        .map {
            key, pairId, platform, segmentConfigJson,
            _runSegmentNuclei, _runSegment ->
                def nucleiMask = requireExistingPath(
                    publishedDatasetPath(
                        params.outdir,
                        pairId,
                        platform,
                        "segmentation/cellpose_nuclei_masks_tiled.npy",
                    ),
                    "Cellpose nuclei mask for ${pairId}:${platform}",
                )
                def nucleiStats = requireExistingPath(
                    publishedDatasetPath(
                        params.outdir,
                        pairId,
                        platform,
                        "segmentation/cellpose_nuclei_stitching_stats.json",
                    ),
                    "Cellpose nuclei stitching stats for ${pairId}:${platform}",
                )
                tuple(
                    key,
                    pairId,
                    platform,
                    segmentConfigJson,
                    nucleiMask,
                    nucleiStats,
                )
        }

    nuclei_results_ch = nuclei_task_results_ch.mix(nuclei_published_results_ch)
    proseg_trigger_ch = cellpose_segment_inputs_ch
        .map { true }
        .mix(atera_proseg_plan_inputs_ch.map { true })
        .mix(atera_proseg_cellpose_plan_inputs_ch.map { true })
        .take(1)
    proseg_path_ch = ENSURE_PROSEG(proseg_trigger_ch)
    segment_task_results_ch = SEGMENT(
        cellpose_segment_inputs_ch,
        nuclei_results_ch,
        proseg_path_ch,
    )

    // A failed tile leaves its group incomplete: groupTuple drops it, so the
    // row's stitch (and its later stages) does not run; the failed tiles are
    // named below.
    atera_proseg_tile_results_ch = ATERA_PROSEG_TILE(
        atera_proseg_tile_specs_ch.combine(proseg_path_ch)
    ).tile

    atera_proseg_tile_specs_ch
        .map { key, pairId, platform, _nTiles, tileId, _tileConfig ->
            tuple(key, "${pairId}:${platform}".toString(), tileId)
        }
        .groupTuple(by: [0, 1])
        .join(
            atera_proseg_tile_results_ch
                .map { key, _nTiles, tileId, _tileDir -> tuple(key, tileId) }
                .groupTuple(),
            remainder: true,
        )
        .subscribe { key, label, expectedIds, finishedIds ->
            if (key == null || label == null) {
                return
            }
            def missing = (expectedIds - (finishedIds ?: [])).sort()
            if (missing) {
                log.error(
                    "${label}: ${missing.size()} of ${expectedIds.size()} ATERA " +
                    "ProSeg tiles failed (${missing.join(', ')}); the ProSeg stitch " +
                    "and every later stage of this row (original_seg and proseg " +
                    "layers) do not run. Fix the cause and re-run with -resume; " +
                    "the finished tiles are cached."
                )
            }
        }

    atera_proseg_tiles_ch = atera_proseg_tile_results_ch
        .map { key, nTiles, tileId, tileDir ->
            tuple(groupKey(key, nTiles), tileId, tileDir)
        }
        .groupTuple()
        .map { groupedKey, tileIds, tileDirs ->
            // Tiles finish in any order; sorting keeps the stitch inputs (and
            // its task hash) stable across runs.
            def pairs = [tileIds, tileDirs].transpose().sort { pair -> pair[0].toString() }
            tuple(
                groupedKey.getGroupTarget(),
                pairs.collect { pair -> pair[0] },
                pairs.collect { pair -> pair[1] },
            )
        }

    // A tile subset is a check run of single tiles: it is never stitched.
    if (ateraProsegTileSubset(params) != null) {
        log.info(
            "atera_proseg_tile_subset=${ateraProsegTileSubset(params).join(',')}: " +
            "the ATERA ProSeg tiles run as a check and are not stitched"
        )
    }
    atera_proseg_stitch_inputs_ch = native_routed_ch.proseg
        .join(atera_proseg_plan_ch)
        .join(atera_proseg_tiles_ch)
        .filter { ateraProsegTileSubset(params) == null }
        .map {
            key, pairId, platform, latestZarr, _maskPath, _transcriptsCsv,
            _nucleiMaskPath, rowSource, _planPairId, _planPlatform, planFile, tileIds,
            tileDirs ->
                tuple(
                    key,
                    pairId,
                    platform,
                    groovy.json.JsonOutput.prettyPrint(
                        groovy.json.JsonOutput.toJson(
                            ateraProsegStitchConfig(
                                pairId,
                                platform,
                                tileIds,
                                params,
                                AteraProsegLayers.priorKindForSource(rowSource),
                            )
                        )
                    ),
                    latestZarr,
                    planFile,
                    tileDirs,
                )
        }

    atera_proseg_stitch_with_stock_ch = atera_proseg_stitch_inputs_ch
        .join(atera_stock_cellpose_ch.map { key, _pair, _platform, mask, _prob, _transform, _stats ->
            tuple(key, mask)
        }, remainder: true)
        .filter { values -> values.size() == 8 && values[1] != null }
        .map { key, pairId, platform, configJson, latest, plan, tileDirs, stockMask ->
            def config = new groovy.json.JsonSlurper().parseText(configJson)
            if (stockMask != null) config.stock_cellpose_mask_path = 'stock_cellpose_mask.npy'
            tuple(key, pairId, platform,
                groovy.json.JsonOutput.prettyPrint(groovy.json.JsonOutput.toJson(config)),
                latest, plan, tileDirs, stockMask ?: [])
        }

    // An ATERA cellpose_proseg row's SEGMENT tuple carries its Cellpose mask in
    // the mask slot (as a XENIUM row's does), for ENRICH's proseg_mask layer
    // and the image quantification; a native_proseg row keeps the placeholder.
    atera_proseg_stitch_ch = ATERA_PROSEG_STITCH(atera_proseg_stitch_with_stock_ch).segmentation
        .join(
            atera_cellpose_ch.map {
                key, _pairId, _platform, cellposeMask, _cellprob, _transforms, _stats ->
                    tuple(key, cellposeMask)
            },
            remainder: true,
        )
        .filter { values -> values.size() == 8 && values[1] != null }
        .map {
            key, pairId, platform, latestZarr, placeholderMask, transcriptsCsv,
            nucleiMaskPath, cellposeMask ->
                tuple(
                    key,
                    pairId,
                    platform,
                    latestZarr,
                    cellposeMask ?: placeholderMask,
                    transcriptsCsv,
                    nucleiMaskPath,
                )
        }

    segment_published_results_ch = sample_rows_ch.flatMap { pairId, _row, settings ->
        if (!(settings.run_enrich && !settings.run_segment)) {
            []
        } else if (runsAteraProseg(settings.segmentation_source, settings.active_platforms)) {
            settings.active_platforms.collect { platform ->
                def key = "${pairId}|${platform}"
                tuple(
                    key,
                    pairId,
                    platform,
                    publishedAteraProsegOutput(params, pairId, platform, "proseg_latest.zarr"),
                    settings.segmentation_source in ["cellpose_proseg", "cellpose_comparison"]
                        ? publishedAteraCellposeMask(params, pairId, platform)
                        : publishedAteraProsegOutput(params, pairId, platform, "NO_CELLPOSE_MASK"),
                    publishedAteraProsegOutput(params, pairId, platform, "NO_TRANSCRIPTS_CSV"),
                    publishedAteraProsegOutput(params, pairId, platform, "NO_NUCLEI_MASK"),
                )
            }
        } else if (settings.segmentation_source == "native") {
            settings.active_platforms.collect { platform ->
                def key = "${pairId}|${platform}"
                tuple(
                    key,
                    pairId,
                    platform,
                    publishedNativeOutput(params, pairId, platform, "native_latest.zarr"),
                    publishedNativeOutput(params, pairId, platform, "NO_CELLPOSE_MASK"),
                    publishedNativeOutput(params, pairId, platform, "NO_TRANSCRIPTS_CSV"),
                    publishedNativeOutput(params, pairId, platform, "NO_NUCLEI_MASK"),
                )
            }
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
                def nucleiMaskPath = requireExistingPath(
                    publishedDatasetPath(
                        params.outdir,
                        pairId,
                        platform,
                        "segmentation/cellpose_nuclei_masks_tiled.npy",
                    ),
                    "SEGMENT Cellpose nuclei mask for ${pairId}:${platform}",
                )
                tuple(
                    key,
                    pairId,
                    platform,
                    latestZarr,
                    maskPath,
                    transcriptsCsv,
                    nucleiMaskPath,
                )
            }
        }
    }

    segment_results_ch = segment_task_results_ch
        .mix(native_plain_ch)
        .mix(atera_proseg_stitch_ch)
        .mix(segment_published_results_ch)

    metadata_ch = build_results_ch.map { key, pairId, platform, sourceSpatialdata ->
        tuple(key, pairId, platform, sourceSpatialdata.toString())
    }

    enrich_gate_ch = sample_rows_ch.flatMap { pairId, _row, settings ->
        if (!settings.run_enrich) {
            []
        } else {
            settings.active_platforms.collect { platform ->
                tuple("${pairId}|${platform}", settings.segmentation_source)
            }
        }
    }

    enrich_inputs_ch = segment_results_ch
        .join(metadata_ch)
        .join(enrich_gate_ch)
        .map {
            key, pairId, platform, latestZarr, maskPath, _transcriptsCsv,
            nucleiMaskPath,
            pairMeta, platformMeta, originalDataPath, rowSource ->
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
                nuclei_mask_path: "enrich_input_nuclei_mask.npy",
                original_data_path: originalDataPath,
                output_dir: "enrich_out",
                persistent_output_path: persistentLatestZarrPath,
                transform_path: null,
            ]
            // Native rows enrich the vendor segmentation; no Cellpose mask is
            // read (the mask slots hold NATIVE_LATEST placeholders). Every
            // ATERA row has a NATIVE_LATEST store, so the vendor enrichment
            // runs for each; the row's source (carried by the gate) adds the
            // rest.
            if (PlatformSpec.forName(platform).nativeLatestBase) {
                enrichConfig.segmentation_source = "native"
                enrichConfig.transcript_density_source_path = originalDataPath
                enrichConfig.mask_path = null
                enrichConfig.nuclei_mask_path = null
            }
            // A native_proseg row also checks its stitched ProSeg layer (AT-9).
            if (rowSource == "native_proseg") {
                enrichConfig.segmentation_source = "native_proseg"
            }
            // An ATERA cellpose_proseg row (AT-7) checks its stitched ProSeg
            // layer (reseg) and builds its Cellpose layer (proseg_mask) from
            // the Cellpose mask in the mask slot; it has no Cellpose nuclei.
            if (rowSource in ["cellpose_proseg", "cellpose_comparison"] && PlatformSpec.forName(platform).nativeLatestBase) {
                enrichConfig.segmentation_source = rowSource
                enrichConfig.mask_path = "enrich_input_mask.npy"
                if (rowSource == 'cellpose_comparison') {
                    enrichConfig.stock_cellpose_mask_path = publishedDatasetPath(
                        params.outdir, pairId, platform,
                        'segmentation/stock_cellpose/cellpose_masks_tiled.npy').toString()
                }
            }

            tuple(
                key,
                pairId,
                platform,
                groovy.json.JsonOutput.prettyPrint(groovy.json.JsonOutput.toJson(enrichConfig)),
                latestZarr,
                maskPath,
                nucleiMaskPath,
            )
        }

    enrich_task_results_ch = ENRICH(enrich_inputs_ch)

    enrich_published_results_ch = sample_rows_ch.flatMap { pairId, row, settings ->
        if (!(settings.need_enriched_zarrs && !settings.run_enrich)) {
            []
        } else {
            settings.active_platforms.collect { platform ->
                def key = "${pairId}|${platform}"
                // On native rows that field names the source store, not a latest
                // store, so the published latest store is always used.
                def explicitLatestZarr = (
                    PlatformSpec.forName(platform).defaultSegmentationSource == "native"
                )
                    ? null
                    : chooseField(
                        row,
                        PlatformSpec.forName(platform).spatialdataPathFields
                    )
                // Cross-pair restarts may intentionally combine durable latest
                // zarrs that live under different source pair IDs.
                def latestZarrPath = isBlankPath(explicitLatestZarr)
                    ? publishedDatasetPath(
                        params.outdir,
                        pairId,
                        platform,
                        "latest/latest_spatialdata.zarr",
                    )
                    : explicitLatestZarr
                def latestZarr = requireExistingPath(
                    latestZarrPath,
                    "ENRICH latest zarr for ${pairId}:${platform}",
                )
                // The published latest zarr is the only ENRICH artifact consumed
                // downstream. Older completed runs may not retain enrich_out, so
                // use the required latest zarr as the unused provenance placeholder
                // instead of rejecting an otherwise valid downstream stage start.
                def enrichOut = latestZarr
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

    // Register externally acquired images (e.g. post-Xenium IF) onto the Xenium
    // mask grid. This writer of the shared store runs before the viewer cache and
    // mask image quantification, which then pick up the new image channels.
    register_images_inputs_ch = enriched_zarrs_ch
        .join(
            sample_rows_ch.flatMap { pairId, _row, settings ->
                settings.run_register_images
                    ? [
                        tuple(
                            "${pairId}|XENIUM",
                            settings.registered_images_csv,
                            settings.register_post_xenium_if_nonrigid,
                        )
                    ]
                    : []
            }
        )
        .map {
            key, pairId, platform, enrichedLatestZarr, registeredImagesCsv,
            registerPostXeniumIfNonrigid ->
            def registerImagesConfig = [
                dataset_name: "${pairId}_${platform}",
                platform: platform,
                latest_zarr_path: "latest_input.zarr",
                output_dir: "register_images_out",
                images: registeredImageSpecs(registeredImagesCsv),
                refine_affine: params.register_images_refine_affine,
                nonrigid_enabled: registerPostXeniumIfNonrigid,
                nonrigid_pixel_size_um: params.register_post_xenium_if_nonrigid_pixel_size_um,
                nonrigid_attachment: params.register_post_xenium_if_nonrigid_attachment,
                nonrigid_tightness: params.register_post_xenium_if_nonrigid_tightness,
                nonrigid_num_warp: params.register_post_xenium_if_nonrigid_num_warp,
                nonrigid_num_iter: params.register_post_xenium_if_nonrigid_num_iter,
                nonrigid_smoothing_sigma_um: params.register_post_xenium_if_nonrigid_smoothing_sigma_um,
                nonrigid_max_displacement_um: params.register_post_xenium_if_nonrigid_max_displacement_um,
                nonrigid_max_cycle_error_um: params.register_post_xenium_if_nonrigid_max_cycle_error_um,
                nonrigid_min_local_correlation: params.register_post_xenium_if_nonrigid_min_local_correlation,
                nonrigid_min_jacobian: params.register_post_xenium_if_nonrigid_min_jacobian,
                nonrigid_max_jacobian: params.register_post_xenium_if_nonrigid_max_jacobian,
                nonrigid_min_confident_fraction: params.register_post_xenium_if_nonrigid_min_confident_fraction,
                nonrigid_residual_passes: params.register_post_xenium_if_nonrigid_residual_passes,
                nonrigid_min_residual_correlation_improvement: params.register_post_xenium_if_nonrigid_min_residual_correlation_improvement,
                nonrigid_min_residual_initial_confidence: params.register_post_xenium_if_nonrigid_min_residual_initial_confidence,
                registration_quality_enabled: params.register_post_xenium_if_registration_quality,
                registration_quality_pixel_size_um: params.register_post_xenium_if_registration_quality_pixel_size_um,
                registration_quality_smoothing_sigma_um: params.register_post_xenium_if_registration_quality_smoothing_sigma_um,
                build_viewer_pyramid: params.viewer_cache_build_image_pyramid,
            ]

            tuple(
                key,
                pairId,
                platform,
                groovy.json.JsonOutput.prettyPrint(groovy.json.JsonOutput.toJson(registerImagesConfig)),
                enrichedLatestZarr,
            )
        }

    register_images_results_ch = REGISTER_IMAGES(register_images_inputs_ch)

    registered_zarrs_ch = register_images_results_ch.map {
        key, pairId, platform, registeredLatestZarr, _registerImagesOutDir ->
            tuple(
                key,
                pairId,
                platform,
                java.nio.file.Paths.get(registeredLatestZarr.toString()).toRealPath().toString(),
            )
    }

    register_images_passthrough_ch = enriched_zarrs_ch
        .join(
            sample_rows_ch.flatMap { pairId, _row, settings ->
                settings.active_platforms
                    .findAll { platform ->
                        settings.need_enriched_zarrs &&
                            !(settings.run_register_images && platform == "XENIUM")
                    }
                    .collect { platform -> tuple("${pairId}|${platform}", true) }
            }
        )
        .map { key, pairId, platform, enrichedLatestZarr, _runFlag ->
            tuple(key, pairId, platform, enrichedLatestZarr)
        }

    post_registration_zarrs_ch = registered_zarrs_ch.mix(register_images_passthrough_ch)

    // Pre-build the napari viewer's derived caches (label masks + label/outline
    // pyramids + image pyramid) into the enriched latest zarr. This is a separate
    // writer of the shared store, so it is serialized into the post-enrich chain:
    // the stages that consume the enriched zarr read `post_enrich_zarrs_ch` below,
    // which is the viewer-cache output when the stage runs and the enriched zarr
    // passthrough otherwise.
    viewer_cache_gate_ch = sample_rows_ch.flatMap { pairId, _row, settings ->
        if (!settings.run_build_viewer_caches) {
            []
        } else {
            settings.active_platforms.collect { platform ->
                tuple("${pairId}|${platform}", true)
            }
        }
    }

    viewer_cache_transform_ch = sample_rows_ch.flatMap { pairId, row, settings ->
        if (!settings.run_build_viewer_caches) {
            []
        } else {
            settings.active_platforms.collect { platform ->
                def transformPath = chooseField(
                    row,
                    PlatformSpec.forName(platform).transformPathFields
                )
                tuple("${pairId}|${platform}", transformPath)
            }
        }
    }

    viewer_cache_inputs_ch = post_registration_zarrs_ch
        .join(viewer_cache_gate_ch)
        .join(viewer_cache_transform_ch)
        .map { key, pairId, platform, enrichedLatestZarr, _runFlag, transformPath ->
            def resolvedTransform = (transformPath && transformPath.toString().trim())
                ? file(transformPath).toAbsolutePath().toString()
                : null
            def viewerCacheConfig = [
                dataset_name: "${pairId}_${platform}",
                platform: platform,
                latest_zarr_path: "latest_input.zarr",
                original_data_path: "latest_input.zarr",
                output_dir: "viewer_cache_out",
                transform_path: resolvedTransform,
                downsample: params.viewer_cache_downsample,
                label_chunk_size: params.viewer_cache_label_chunk_size,
                contour_width: params.viewer_cache_contour_width,
                min_size: params.viewer_cache_min_size,
                build_image_pyramid: params.viewer_cache_build_image_pyramid,
            ]
            // Native rows draw the vendor label masks on the native image.
            if (PlatformSpec.forName(platform).defaultSegmentationSource == "native") {
                viewerCacheConfig.base_image_key = PlatformSpec.forName(platform).nativeImageKey
            }

            tuple(
                key,
                pairId,
                platform,
                groovy.json.JsonOutput.prettyPrint(groovy.json.JsonOutput.toJson(viewerCacheConfig)),
                enrichedLatestZarr,
            )
        }

    viewer_cache_results_ch = VIEWER_CACHE(viewer_cache_inputs_ch)

    viewer_cached_zarrs_ch = viewer_cache_results_ch.map {
        key, pairId, platform, cachedLatestZarr, _viewerCacheOutDir ->
            tuple(
                key,
                pairId,
                platform,
                java.nio.file.Paths.get(cachedLatestZarr.toString()).toRealPath().toString(),
            )
    }

    viewer_cache_passthrough_gate_ch = sample_rows_ch.flatMap { pairId, _row, settings ->
        if (!(settings.need_enriched_zarrs && !settings.run_build_viewer_caches)) {
            []
        } else {
            settings.active_platforms.collect { platform ->
                tuple("${pairId}|${platform}", true)
            }
        }
    }

    viewer_cache_passthrough_ch = post_registration_zarrs_ch
        .join(viewer_cache_passthrough_gate_ch)
        .map { key, pairId, platform, enrichedLatestZarr, _runFlag ->
            tuple(key, pairId, platform, enrichedLatestZarr)
        }

    post_enrich_zarrs_ch = viewer_cached_zarrs_ch.mix(viewer_cache_passthrough_ch)

    mask_image_quantification_gate_ch = sample_rows_ch.flatMap {
        pairId, _row, settings ->
            if (!settings.run_mask_image_quantification) {
                []
            } else {
                settings.active_platforms.collect { platform ->
                    tuple("${pairId}|${platform}", settings.segmentation_source)
                }
            }
    }

    mask_image_quantification_task_masks_ch = segment_task_results_ch
        .mix(native_plain_ch)
        .mix(atera_proseg_stitch_ch)
        .map {
            key, _pairId, _platform, _latestZarr, maskPath, _transcriptsCsv,
            _nucleiMaskPath ->
                tuple(key, maskPath)
        }

    mask_image_quantification_published_masks_ch = sample_rows_ch.flatMap {
        pairId, _row, settings ->
            if (!(
                settings.run_mask_image_quantification &&
                !settings.run_segment
            )) {
                []
            } else {
                settings.active_platforms.collect { platform ->
                    def key = "${pairId}|${platform}"
                    // Native rows quantify the vendor label element; the slot
                    // holds the NATIVE_LATEST placeholder.
                    def maskPath = settings.segmentation_source == "native_proseg"
                        ? publishedAteraProsegOutput(params, pairId, platform, "NO_CELLPOSE_MASK")
                        : settings.segmentation_source == "native"
                        ? publishedNativeOutput(params, pairId, platform, "NO_CELLPOSE_MASK")
                        : requireExistingPath(
                            publishedDatasetPath(
                                params.outdir,
                                pairId,
                                platform,
                                "segmentation/cellpose_masks_tiled.npy",
                            ),
                            "SEGMENT Cellpose mask for ${pairId}:${platform}",
                        )
                    tuple(key, maskPath)
                }
            }
    }

    mask_image_quantification_masks_ch =
        mask_image_quantification_task_masks_ch.mix(
            mask_image_quantification_published_masks_ch
        )

    mask_image_quantification_inputs_ch = post_enrich_zarrs_ch
        .join(mask_image_quantification_masks_ch)
        .join(mask_image_quantification_gate_ch)
        .map { key, pairId, platform, enrichedLatestZarr, maskPath, rowSource ->
            def quantConfig = [
                dataset_name: "${pairId}_${platform}",
                platform: platform,
                latest_zarr_path: "latest_input.zarr",
                mask_path: "mask_image_quantification_input_mask.npy",
                output_dir: "mask_image_quantification_out",
            ]
            // Native rows quantify the vendor cell labels of the store; the
            // linked mask slot is an empty placeholder.
            def quantSpec = PlatformSpec.forName(platform)
            if (quantSpec.nativeLatestBase) {
                quantConfig.mask_path = null
                quantConfig.mask_label_key = "cell_labels"
                quantConfig.shape_key = quantSpec.originalCellShapeKey
                quantConfig.table_key =
                    "table_${quantSpec.originalCellShapeKey}_image_quantification"
            }
            // A tiled ProSeg row also quantifies its stitched ProSeg cells in
            // the same task (one writer per store), by their label element; an
            // ATERA cellpose_proseg row also its Cellpose cells (the label
            // element ENRICH writes: the global frame of an ATERA store is in
            // pixels, so every ATERA layer is quantified from labels).
            def prosegPriorKind = quantSpec.nativeLatestBase
                ? AteraProsegLayers.priorKindForSource(rowSource)
                : null
            if (prosegPriorKind != null) {
                def prosegLayer = AteraProsegLayers.forKind(prosegPriorKind)
                quantConfig.extra_layers = [
                    [
                        mask_label_key: prosegLayer.label_key,
                        shape_key: prosegLayer.shapes_key,
                        table_key: "${prosegLayer.table_key}_image_quantification",
                    ],
                ]
                if (prosegPriorKind == "cellpose") {
                    quantConfig.extra_layers << [
                        mask_label_key: "cellpose_cell_labels",
                        shape_key: "MOSAIK_cellpose",
                        table_key: "table_MOSAIK_cellpose_image_quantification",
                    ]
                    if (rowSource == 'cellpose_comparison') {
                        quantConfig.extra_layers << [
                            mask_label_key: 'stock_cellpose_cell_labels',
                            shape_key: 'MOSAIK_stock_cellpose',
                            table_key: 'table_MOSAIK_stock_cellpose_image_quantification',
                        ]
                    }
                }
            }

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

    enriched_downstream_zarrs_ch = post_enrich_zarrs_ch
        .join(enriched_downstream_gate_ch)
        .map { key, pairId, platform, enrichedLatestZarr, _runFlag ->
            tuple(key, pairId, platform, enrichedLatestZarr)
        }

    downstream_zarrs_ch = enriched_downstream_zarrs_ch.mix(quantified_zarrs_ch)

    // Cortical depth now runs after clustering_squidpy (wired below) so that the
    // per-cell broad_class / subcluster_label annotations exist and the depth
    // violin plots can be produced. The analysis branch therefore reads the
    // enriched/quantified zarrs directly and does not wait on cortical depth,
    // which also breaks what would otherwise be a cycle:
    // clustering -> cortical depth -> clustering.
    analysis_ready_zarrs_ch = downstream_zarrs_ch

    analysis_layer_validation_gate_ch = sample_rows_ch.flatMap {
        pairId, _row, settings ->
        if (!settings.need_analysis_zarrs) {
            []
        } else {
            settings.active_platforms.collect { platform ->
                tuple("${pairId}|${platform}", settings)
            }
        }
    }

    analysis_layer_validation_inputs_ch = analysis_ready_zarrs_ch
        .join(analysis_layer_validation_gate_ch)
        .flatMap {
            key, pairId, platform, enrichedLatestZarr, settings ->
            settings.analysis_input_segmentations.collect { segmentation ->
                def layerKeys = analysisLayerKeys(platform, segmentation)
                tuple(
                    "${key}|${segmentation}",
                    pairId,
                    platform,
                    segmentation,
                    enrichedLatestZarr,
                    layerKeys.table_key,
                    layerKeys.shape_key,
                    settings,
                )
            }
        }

    analysis_layer_validation_results_ch = VALIDATE_ANALYSIS_LAYER(
        analysis_layer_validation_inputs_ch
    )

    qc_inputs_ch = analysis_layer_validation_results_ch
        .filter {
            _key, _pairId, _platform, _segmentation, _latestZarr,
            _tableKey, _shapeKey, settings, _validationJson ->
                settings.run_qc && settings.analysis_segmentations.contains(_segmentation)
        }
        .map {
            key, pairId, platform, segmentation, latestZarr,
            tableKey, shapeKey, _settings, _validationJson ->
                tuple(
                    key,
                    pairId,
                    platform,
                    segmentation,
                    latestZarr,
                    tableKey,
                    shapeKey,
                )
        }

    qc_results_ch = QC(qc_inputs_ch)

    analysis_branch_settings_ch = sample_rows_ch.flatMap { pairId, _row, settings ->
        if (!settings.need_analysis_zarrs) {
            []
        } else {
            settings.active_platforms.collectMany { platform ->
                settings.analysis_input_segmentations.collect { segmentation ->
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

    analysis_without_qc_ch = analysis_layer_validation_results_ch
        .filter {
            _key, _pairId, _platform, _segmentation, _latestZarr,
            _tableKey, _shapeKey, settings, _validationJson ->
                !settings.run_qc ||
                    !settings.analysis_segmentations.contains(_segmentation)
        }
        .map {
            _key, pairId, platform, segmentation, latestZarr,
            tableKey, shapeKey, settings, _validationJson ->
                    tuple(
                        pairId,
                        segmentation,
                        platform,
                        latestZarr,
                        tableKey,
                        shapeKey,
                        settings,
                    )
        }

    analysis_dataset_zarrs_ch = analysis_from_qc_ch.mix(analysis_without_qc_ch)

    mecr_samples_ch = analysis_dataset_zarrs_ch
        .filter {
            _pairId, _segmentation, _platform, _zarrPath, _tableKey, _shapeKey,
            settings ->
                settings.run_mecr &&
                    settings.analysis_segmentations.contains(_segmentation)
        }
        .map {
            pairId, segmentation, platform, zarrPath, _tableKey, _shapeKey,
            settings ->
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
        .map {
            _branchKey, pairIds, segmentations, settingsList, platformNames,
            zarrPaths ->
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
                )
        }

    // XENIUM and MERSCOPE rows share one reference over their merged panels
    // (MECR_REFERENCE, its JSON unchanged by ATERA rows). ATERA rows get their
    // own streaming reference over the whole-transcriptome panel (Atera plan
    // AT-8).
    mecr_samples_by_reference_ch = mecr_samples_ch
        .branch { _pairId, _segmentation, samplesJson ->
            atera: mecrSamplesUseAteraReference(samplesJson)
            standard: true
        }
    mecr_reference_inputs_ch = mecrReferenceInputs(mecr_samples_by_reference_ch.standard)
    mecr_reference_results_ch = MECR_REFERENCE(mecr_reference_inputs_ch)
    mecr_atera_reference_inputs_ch = mecrReferenceInputs(
        mecr_samples_by_reference_ch.atera
    )
    mecr_atera_reference_results_ch = MECR_REFERENCE_ATERA(
        mecr_atera_reference_inputs_ch
    )
    mecr_inputs_ch = mecr_samples_by_reference_ch.standard
        .combine(mecr_reference_results_ch)
        .map { pairId, segmentation, samplesJson, referenceOut ->
            tuple(pairId, segmentation, samplesJson, referenceOut)
        }
        .mix(
            mecr_samples_by_reference_ch.atera
                .combine(mecr_atera_reference_results_ch)
                .map { pairId, segmentation, samplesJson, referenceOut ->
                    tuple(pairId, segmentation, samplesJson, referenceOut)
                }
        )
    mecr_results_ch = MECR(mecr_inputs_ch)

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
        .map { pairId, merscopePath, xeniumPath, settings ->
            def isValis = params.alignment_backend
                .toString()
                .trim()
                .toLowerCase() == "valis"
            def legacyPlaceholder = params.samplesheet
            tuple(
                pairId,
                merscopePath,
                xeniumPath,
                file(
                    isValis
                        ? settings.alignment_annotation_paths.MERSCOPE
                        : legacyPlaceholder
                ),
                file(
                    isValis
                        ? settings.alignment_annotation_paths.XENIUM
                        : legacyPlaceholder
                ),
            )
        }

    alignment_registration_results_ch = ALIGN(align_inputs_ch)
    alignment_task_results_ch = MATERIALIZE_ALIGNMENT(alignment_registration_results_ch)
        .map { pairId, merscopePath, xeniumPath, alignOut, _materializationSummary ->
            tuple(
                pairId,
                merscopePath,
                xeniumPath,
                alignOut.resolve("alignment_transform.json"),
                alignOut.resolve("alignment_coords"),
                alignOut,
            )
        }

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
                    transformJson.parent,
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
        .map { pairId, merscopeLatest, xeniumLatest, _transformJson, _coordsDir, alignOut, _runFlag ->
            tuple(pairId, merscopeLatest, xeniumLatest, alignOut)
        }

    alignment_qc_results_ch = ALIGN_QC(align_qc_inputs_ch)

    alignment_done_no_qc_gate_ch = sample_rows_ch.flatMap { pairId, _row, settings ->
        if (!(settings.need_alignment_downstream && !settings.run_align_qc)) {
            []
        } else {
            [tuple(pairId, settings.analysis_input_segmentations)]
        }
    }

    alignment_done_no_qc_ch = alignment_results_ch
        .join(alignment_done_no_qc_gate_ch)
        .flatMap {
            pairId, _merscopeLatest, _xeniumLatest, _transformJson, _coordsDir,
            _alignOut, analysisSegmentations ->
                analysisSegmentations.collect { segmentation ->
                    tuple("${pairId}|${segmentation}", true)
                }
        }

    alignment_done_after_qc_gate_ch = sample_rows_ch.flatMap { pairId, _row, settings ->
        if (!(settings.need_alignment_downstream && settings.run_align_qc)) {
            []
        } else {
            [tuple(pairId, settings.analysis_input_segmentations)]
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
                     settings.run_spatial_gene_analysis ||
                     settings.run_clustering_squidpy)
        }

    paired_downstream_align_ch = paired_analysis_zarrs_ch
        .filter {
            _pairId, _segmentation, _merscopePath, _xeniumPath, _merscopeTableKey,
            _merscopeShapeKey, _xeniumTableKey, _xeniumShapeKey, settings ->
                settings.enable_alignment &&
                    (settings.run_compare ||
                     settings.run_visualize ||
                     settings.run_spatial_gene_analysis ||
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
                settings.run_compare &&
                    settings.analysis_segmentations.contains(_segmentation)
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
                    (
                        settings.run_visualize ||
                        settings.run_spatial_gene_analysis ||
                        settings.run_clustering_squidpy
                    )
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
                    (
                        settings.run_visualize ||
                        settings.run_spatial_gene_analysis ||
                        settings.run_clustering_squidpy
                    )
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
        .filter { _pairId, segmentation, _samplesJson, settings ->
            settings.analysis_segmentations.contains(segmentation) &&
                settings.run_visualize && !settings.run_compare
        }
        .map { pairId, segmentation, samplesJson, _settings ->
            tuple(pairId, segmentation, samplesJson)
        }

    visualize_after_compare_ch = analysis_samples_ch
        .filter { _pairId, segmentation, _samplesJson, settings ->
            settings.analysis_segmentations.contains(segmentation) &&
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

    spatial_gene_analysis_without_visualize_ch = analysis_samples_ch
        .filter { _pairId, segmentation, _samplesJson, settings ->
            settings.analysis_segmentations.contains(segmentation) &&
                settings.run_spatial_gene_analysis && !settings.run_visualize
        }
        .map { pairId, segmentation, samplesJson, settings ->
            tuple(
                pairId,
                segmentation,
                samplesJson,
                settings.spatial_gene_analysis_transcript_analysis_enabled,
            )
        }

    spatial_gene_analysis_after_visualize_ch = analysis_samples_ch
        .filter { _pairId, segmentation, _samplesJson, settings ->
            settings.analysis_segmentations.contains(segmentation) &&
                settings.run_spatial_gene_analysis && settings.run_visualize
        }
        .map { pairId, segmentation, samplesJson, settings ->
            tuple(
                "${pairId}|${segmentation}",
                pairId,
                segmentation,
                samplesJson,
                settings.spatial_gene_analysis_transcript_analysis_enabled,
            )
        }
        .join(visualize_done_ch)
        .map {
            _branchKey,
            pairId,
            segmentation,
            samplesJson,
            transcriptAnalysisEnabled,
            _doneFlag ->
            tuple(
                pairId,
                segmentation,
                samplesJson,
                transcriptAnalysisEnabled,
            )
        }

    spatial_gene_analysis_unannotated_inputs_ch =
        spatial_gene_analysis_without_visualize_ch.mix(
            spatial_gene_analysis_after_visualize_ch
        )

    spatial_gene_annotation_rows_ch = sample_rows_ch.flatMap {
        pairId, row, settings ->
            if (!settings.run_spatial_gene_analysis) {
                []
            } else {
                settings.analysis_segmentations.collect { segmentation ->
                    tuple("${pairId}|${segmentation}", row)
                }
            }
    }

    spatial_gene_analysis_inputs_ch = spatial_gene_analysis_unannotated_inputs_ch
        .map { pairId, segmentation, samplesJson, transcriptAnalysisEnabled ->
            tuple(
                "${pairId}|${segmentation}",
                pairId,
                segmentation,
                samplesJson,
                transcriptAnalysisEnabled,
            )
        }
        .join(spatial_gene_annotation_rows_ch)
        .map {
            _key,
            pairId,
            segmentation,
            samplesJson,
            transcriptAnalysisEnabled,
            row ->
            tuple(
                pairId,
                segmentation,
                spatialGeneSamplesJson(samplesJson, row),
                transcriptAnalysisEnabled,
            )
        }

    spatial_gene_analysis_results_ch = SPATIAL_GENE_ANALYSIS(
        spatial_gene_analysis_inputs_ch
    )

    spatial_gene_analysis_done_ch = spatial_gene_analysis_results_ch.map {
        pairId, segmentation, _spatialGeneAnalysisOutDir ->
            tuple("${pairId}|${segmentation}", true)
    }

    clustering_without_dependencies_ch = analysis_samples_ch
        .filter { _pairId, segmentation, _samplesJson, settings ->
            settings.run_clustering_squidpy &&
                (
                    !settings.analysis_segmentations.contains(segmentation) ||
                    (!settings.run_visualize && !settings.run_spatial_gene_analysis)
                )
        }
        .map { pairId, segmentation, samplesJson, _settings ->
            tuple(pairId, segmentation, samplesJson)
        }

    clustering_after_visualize_ch = analysis_samples_ch
        .filter { _pairId, segmentation, _samplesJson, settings ->
            settings.run_clustering_squidpy &&
                settings.analysis_segmentations.contains(segmentation) &&
                settings.run_visualize &&
                !settings.run_spatial_gene_analysis
        }
        .map { pairId, segmentation, samplesJson, _settings ->
            tuple("${pairId}|${segmentation}", pairId, segmentation, samplesJson)
        }
        .join(visualize_done_ch)
        .map { _branchKey, pairId, segmentation, samplesJson, _doneFlag ->
            tuple(pairId, segmentation, samplesJson)
        }

    clustering_after_spatial_gene_analysis_ch = analysis_samples_ch
        .filter { _pairId, segmentation, _samplesJson, settings ->
            settings.run_clustering_squidpy &&
                settings.analysis_segmentations.contains(segmentation) &&
                settings.run_spatial_gene_analysis
        }
        .map { pairId, segmentation, samplesJson, _settings ->
            tuple("${pairId}|${segmentation}", pairId, segmentation, samplesJson)
        }
        .join(spatial_gene_analysis_done_ch)
        .map { _branchKey, pairId, segmentation, samplesJson, _doneFlag ->
            tuple(pairId, segmentation, samplesJson)
        }

    clustering_inputs_ch =
        clustering_without_dependencies_ch
            .mix(clustering_after_visualize_ch)
            .mix(clustering_after_spatial_gene_analysis_ch)

    // rca-hook:H5: map_first clustering (plan §2.4, §3.1). The mode is one per
    // row (its clustering_squidpy_mode column, else the run's; §20 D15). With
    // any map_first row, rows are prepared once (a legacy row's PREPARE hash
    // is unchanged) and legacy rows go on to the legacy COMPUTE; a run whose
    // rows are all legacy runs exactly the legacy PREPARE -> COMPUTE
    // statements of the else-branch, keeping its DAG and task hashes.
    // CLUSTERING_MAP_FIRST emits FINALIZE's input, so FINALIZE, the MENDER
    // barrier and every downstream stage are shared by both modes.
    if (AnnotationSettings.isMapFirstRun(params)) {
        row_is_map_first_ch = sample_rows_ch.map { pairId, _row, settings ->
            tuple(pairId, AnnotationSettings.isMapFirst(settings))
        }
        clustering_inputs_ch
            .combine(row_is_map_first_ch, by: 0)
            .filter { _pairId, _segmentation, _samplesJson, mapFirst -> mapFirst }
            .subscribe { pairId, segmentation, _samplesJson, _mapFirst ->
                AnnotationRunRecord.expect(pairId, segmentation)
            }
        clustering_prepared_ch = CLUSTERING_SQUIDPY_PREPARE(
            clustering_inputs_ch
                .combine(sample_rows_ch, by: 0)
                .map { pairId, segmentation, samplesJson, row, settings ->
                    tuple(
                        pairId,
                        segmentation,
                        AnnotationSettings.samplesJsonWithRowColumns(samplesJson, row, settings),
                    )
                }
        )
        clustering_prepared_by_mode_ch = clustering_prepared_ch
            .combine(row_is_map_first_ch, by: 0)
            .branch { _pairId, _segmentation, _samplesJson, _config, _preparedDir, mapFirst ->
                map_first: mapFirst
                legacy: true
            }
        // The registration checks RESOLVE reads (G1: the mouse gate's, the
        // human real-data QC's NR9; M13 C15): per pair x segmentation, the
        // *_registration_qc.json of each active platform. A branch whose QC
        // stage runs in this launch (qc_inputs_ch's filter: run_qc and an
        // analysis segmentation) takes its own QC tasks' checks, released
        // once all of them have finished (groupKey), and never a published
        // one, even when its QC wrote none. Every branch of a row whose QC
        // stage does not run at all (settings.run_qc false: a start_stage
        // after qc or an only_stage, the run's or the row's) takes the checks
        // an earlier launch's QC published under params.outdir for the
        // outputs this launch starts from, or none. A segmentation that is
        // not an analysis segmentation of a row whose QC runs (a MENDER-only
        // one) takes none and never a published one: the launch's earlier
        // stages may have re-segmented it, so a published check could be an
        // older launch's. Without a check, mouse RESOLVE refuses and human G1
        // is not_evaluable. The source of each branch's checks, or why none
        // was looked up, goes to the end-of-run summary (hook H6).
        map_first_registration_from_qc_ch = qc_results_ch
            .map { _key, pairId, _platform, segmentation, _latestZarr, qcOutDir, _tableKey, _shapeKey ->
                tuple(pairId, segmentation, qcOutDir)
            }
            .combine(
                sample_rows_ch.map { pairId, _row, settings ->
                    tuple(pairId, settings.active_platforms.size())
                },
                by: 0,
            )
            .map { pairId, segmentation, qcOutDir, nPlatforms ->
                tuple(
                    groupKey(AnnotationReferences.branchKey(pairId, segmentation), nPlatforms),
                    pairId,
                    segmentation,
                    qcOutDir,
                    nPlatforms,
                )
            }
            .groupTuple(remainder: true)
            .map { _branchKey, pairIds, segmentations, qcOutDirs, nPlatforms ->
                tuple(
                    pairIds[0],
                    segmentations[0],
                    AnnotationReferences.registrationQcFiles(qcOutDirs),
                    nPlatforms[0],
                    AnnotationRunRecord.REGISTRATION_QC_STAGE,
                )
            }
        map_first_registration_published_ch = sample_rows_ch.flatMap { pairId, _row, settings ->
            settings.analysis_input_segmentations
                .findAll { segmentation ->
                    !(settings.run_qc && settings.analysis_segmentations.contains(segmentation))
                }
                .collect { segmentation ->
                    def nPlatforms = settings.active_platforms.size()
                    if (settings.run_qc) {
                        if (AnnotationSettings.isMapFirst(settings)) {
                            log.warn(
                                "${pairId}:${segmentation}: the row's QC stage runs in this launch " +
                                "but not for this segmentation (not an analysis_segmentation), so " +
                                "RESOLVE stages no registration check and published ones are not " +
                                "used (mouse RESOLVE refuses, human registration G1 is not_evaluable)"
                            )
                        }
                        tuple(
                            pairId,
                            segmentation,
                            [],
                            nPlatforms,
                            AnnotationRunRecord.REGISTRATION_ROW_RUNS_QC,
                        )
                    } else {
                        def publishedChecks = AnnotationReferences.publishedRegistrationQcFiles(
                            params.outdir,
                            pairId,
                            settings.active_platforms,
                            segmentation,
                        )
                        if (AnnotationSettings.isMapFirst(settings)) {
                            def branchText = "${pairId}:${segmentation}: the QC stage does not run in this launch"
                            if (publishedChecks) {
                                log.info(
                                    "${branchText}; RESOLVE stages the published registration checks " +
                                    publishedChecks.join(", ")
                                )
                            } else {
                                def searched = AnnotationReferences.publishedQcOutDirs(
                                    params.outdir,
                                    pairId,
                                    settings.active_platforms,
                                    segmentation,
                                )
                                log.warn(
                                    "${branchText} and no registration check is published in " +
                                    "${searched.join(', ')} (mouse RESOLVE refuses, human " +
                                    "registration G1 is not_evaluable)"
                                )
                            }
                        }
                        tuple(
                            pairId,
                            segmentation,
                            publishedChecks,
                            nPlatforms,
                            AnnotationRunRecord.REGISTRATION_PUBLISHED,
                        )
                    }
                }
        }
        map_first_registration_sources_ch = map_first_registration_from_qc_ch
            .mix(map_first_registration_published_ch)
        map_first_registration_sources_ch.subscribe { pairId, segmentation, registrationFiles, nPlatforms, source ->
            AnnotationRunRecord.registrationChecks(pairId, segmentation, source, registrationFiles, nPlatforms)
        }
        map_first_registration_ch = map_first_registration_sources_ch
            .map { pairId, segmentation, registrationFiles, _nPlatforms, _source ->
                tuple(pairId, segmentation, registrationFiles)
            }
        map_first_alignment_ch = alignment_results_ch
            .map { pairId, _merscopeLatest, _xeniumLatest, _transformJson, _coordsDir, alignOut ->
                tuple(pairId, AnnotationReferences.alignmentFiles(alignOut))
            }
            .mix(
                sample_rows_ch
                    .filter { _pairId, _row, settings -> !settings.need_alignment_results }
                    .map { pairId, _row, _settings -> tuple(pairId, []) }
            )
        hippocampal_prepared_ch = clustering_prepared_by_mode_ch.map_first.branch {
            _pairId, _segmentation, samplesJson, _config, _prepared, _mapFirst ->
            hippocampal: new groovy.json.JsonSlurperClassic().parseText(samplesJson).every { sample -> sample.platform == "ATERA" }
            frontal: true
        }
        hippocampal_computed_ch = channel.empty()
        hippocampal_labels_ch = channel.empty()
        if (params.annotation_hippocampal_resources) {
            hippocampal = HIPPOCAMPAL_ANNOTATION(hippocampal_prepared_ch.hippocampal.map { item -> item.take(5) })
            hippocampal_computed_ch = hippocampal.computed
            hippocampal_labels_ch = hippocampal.labels
        }
        clustering_computed_ch = CLUSTERING_MAP_FIRST(
            hippocampal_prepared_ch.frontal.map { item -> item.take(5) },
            map_first_alignment_ch,
            map_first_registration_ch,
        ).computed
            .mix(hippocampal_computed_ch)
            .mix(
                CLUSTERING_SQUIDPY_COMPUTE(
                    clustering_prepared_by_mode_ch.legacy.map { item -> item.take(5) }
                )
            )
    } else {
        clustering_prepared_ch = CLUSTERING_SQUIDPY_PREPARE(clustering_inputs_ch)
        clustering_computed_ch = CLUSTERING_SQUIDPY_COMPUTE(clustering_prepared_ch)
    }
    clustering_results_ch = CLUSTERING_SQUIDPY_FINALIZE(clustering_computed_ch)

    pciseq_reference_inputs_ch = sample_rows_ch
        .filter { _pairId, _row, settings -> settings.run_pciseq }
        .map { _pairId, _row, settings ->
            def reference = settings.pciseq_reference
            tuple(
                settings.pciseq_reference_key,
                groovy.json.JsonOutput.toJson(reference),
                isBlankPath(reference.reference_path) ? [] : [normalizedPath(reference.reference_path)],
                isBlankPath(reference.custom_metadata_path) ? [] : [normalizedPath(reference.custom_metadata_path)],
            )
        }
        .unique { entry -> entry[0] }
    pciseq_reference_results_ch = PCISEQ_REFERENCE(pciseq_reference_inputs_ch)

    pciseq_current_gate_ch = sample_rows_ch.flatMap { pairId, _row, settings ->
        if (!(settings.run_pciseq && settings.run_clustering_squidpy)) {
            []
        } else {
            settings.analysis_segmentations.collect { segmentation ->
                tuple("${pairId}|${segmentation}", settings)
            }
        }
    }

    pciseq_current_inputs_ch = clustering_results_ch
        .map { pairId, segmentation, samplesJson, clusteringOutDir ->
            tuple("${pairId}|${segmentation}", pairId, segmentation, samplesJson, clusteringOutDir)
        }
        .join(pciseq_current_gate_ch)
        .flatMap { _key, pairId, segmentation, samplesJson, clusteringOutDir, settings ->
            def samples = new groovy.json.JsonSlurper().parseText(samplesJson.toString())
            samples.collect { sample ->
                def platform = sample.platform.toString()
                def sampleId = sample.sample_id.toString()
                def h5ad = normalizedPath(clusteringOutDir).resolve(
                    "${platform.toLowerCase()}/${sampleId}_clustered.h5ad",
                )
                def zarr = normalizedPath(sample.zarr_path).toRealPath()
                tuple(
                    settings.pciseq_reference_key,
                    pairId, segmentation, platform,
                    pciseqConfigJson(pairId, segmentation, platform, sampleId, zarr, settings, params),
                    h5ad,
                    isBlankPath(params.pciseq_labels_path) ? [] : [normalizedPath(params.pciseq_labels_path)],
                )
            }
        }

    pciseq_published_inputs_ch = sample_rows_ch.flatMap { pairId, row, settings ->
        if (!(settings.run_pciseq && !settings.run_clustering_squidpy)) {
            []
        } else {
            settings.analysis_segmentations.collectMany { segmentation ->
                settings.active_platforms.collect { platform ->
                    def sampleId = "${pairId}_${platform}"
                    def zarr = pciseqSpatialdataPath(row, settings, platform, params).toRealPath()
                    tuple(
                        settings.pciseq_reference_key,
                        pairId, segmentation, platform,
                        pciseqConfigJson(pairId, segmentation, platform, sampleId, zarr, settings, params),
                        pciseqPublishedH5adPath(pairId, segmentation, platform, params),
                        isBlankPath(params.pciseq_labels_path) ? [] : [normalizedPath(params.pciseq_labels_path)],
                    )
                }
            }
        }
    }

    pciseq_inputs_ch = pciseq_current_inputs_ch
        .mix(pciseq_published_inputs_ch)
        .combine(pciseq_reference_results_ch, by: 0)
        .map {
            _referenceKey, pairId, segmentation, platform, configJson,
            sourceH5ad, labelFiles, referenceBundle ->
                tuple(
                    pairId, segmentation, platform, configJson,
                    sourceH5ad, referenceBundle, labelFiles,
                )
        }
    pciseq_prepared_ch = PCISEQ_PREPARE(pciseq_inputs_ch)
    pciseq_computed_ch = PCISEQ_COMPUTE(pciseq_prepared_ch)
    pciseq_results_ch = PCISEQ_FINALIZE(pciseq_computed_ch)

    pciseq_completion_expectations_ch = sample_rows_ch
        .filter { _pairId, _row, settings -> settings.run_pciseq }
        .map { pairId, _row, settings ->
            tuple(pairId, settings.active_platforms.size() * settings.analysis_segmentations.size())
        }
    pciseq_done_per_pair_ch = pciseq_results_ch
        .map { pairId, _segmentation, _platform, _outDir -> tuple(pairId, true) }
        .combine(pciseq_completion_expectations_ch, by: 0)
        .map { pairId, _done, expectedCount ->
            tuple(groupKey(pairId, expectedCount as int), true)
        }
        .groupTuple()
        .map { pairKey, _doneFlags -> tuple(pairKey.getGroupTarget(), true) }

    // This gate covers both live clustering and published-input restarts.
    // Later stages cannot read the store until every pciSeq writer has finished.
    post_pciseq_sample_rows_ch = sample_rows_ch
        .filter { _pairId, _row, settings -> !settings.run_pciseq }
        .mix(
            sample_rows_ch
                .filter { _pairId, _row, settings -> settings.run_pciseq }
                .join(pciseq_done_per_pair_ch)
                .map { pairId, row, settings, _done -> tuple(pairId, row, settings) }
        )
    post_pciseq_pair_gate_ch = post_pciseq_sample_rows_ch.map {
        pairId, _row, _settings -> tuple(pairId, true)
    }
    clustering_downstream_results_ch = clustering_results_ch
        .combine(post_pciseq_pair_gate_ch, by: 0)
        .map { pairId, segmentation, samplesJson, clusteringOutDir, _done ->
            tuple(pairId, segmentation, samplesJson, clusteringOutDir)
        }

    // Cortical depth runs after clustering so the per-cell broad_class and
    // subcluster_label annotations exist and its depth violin plots can be
    // produced. The distance-from-object branch consumes its tissue-region
    // annotation when both opt-in stages run in the same invocation.
    clustering_completion_expectations_ch = post_pciseq_sample_rows_ch.flatMap {
        pairId, _row, settings ->
            if (!(settings.run_clustering_squidpy &&
                  (settings.run_compute_cortical_depth ||
                   settings.run_distance_from_object))) {
                []
            } else {
                [tuple(pairId, settings.required_clustering_segmentations.size())]
            }
    }

    clustering_done_per_pair_ch = clustering_downstream_results_ch
        .map { pairId, _segmentation, _samplesJson, _clusteringOutDir ->
            tuple(pairId, true)
        }
        .join(clustering_completion_expectations_ch)
        .map { pairId, _done, expectedCount ->
            tuple(groupKey(pairId, expectedCount as int), true)
        }
        .groupTuple()
        .map { pairKey, _doneFlags -> tuple(pairKey.getGroupTarget(), true) }

    compute_cortical_depth_after_clustering_gate_ch =
        post_pciseq_sample_rows_ch.flatMap { pairId, row, settings ->
            if (!(settings.run_compute_cortical_depth && settings.run_clustering_squidpy)) {
                []
            } else {
                def corticalSegmentations = settings.analysis_segmentations
                if (settings.run_distance_from_object) {
                    corticalSegmentations = (
                        corticalSegmentations + settings.distance_from_object_segmentations
                    ).unique()
                }
                settings.active_platforms.collect { platform ->
                    tuple("${pairId}|${platform}", row, corticalSegmentations)
                }
            }
        }

    compute_cortical_depth_after_clustering_ch = downstream_zarrs_ch
        .join(compute_cortical_depth_after_clustering_gate_ch)
        .map { key, pairId, platform, latestZarr, row, analysisSegmentations ->
            tuple(pairId, key, platform, latestZarr, row, analysisSegmentations)
        }
        .combine(clustering_done_per_pair_ch, by: 0)
        .map { pairId, key, platform, latestZarr, row, analysisSegmentations, _done ->
            tuple(key, pairId, platform, latestZarr, row, analysisSegmentations)
        }

    // Fallback for invocations where cortical depth is enabled but clustering is
    // not run in the same pass (e.g. --only_stage compute_cortical_depth after a
    // prior full run): read the enriched/quantified zarr directly. Violins are
    // produced when a clustering table already exists in that zarr, otherwise
    // they are skipped with a logged note.
    compute_cortical_depth_without_clustering_gate_ch =
        post_pciseq_sample_rows_ch.flatMap { pairId, row, settings ->
            if (!(settings.run_compute_cortical_depth && !settings.run_clustering_squidpy)) {
                []
            } else {
                def corticalSegmentations = settings.analysis_segmentations
                if (settings.run_distance_from_object) {
                    corticalSegmentations = (
                        corticalSegmentations + settings.distance_from_object_segmentations
                    ).unique()
                }
                settings.active_platforms.collect { platform ->
                    tuple("${pairId}|${platform}", row, corticalSegmentations)
                }
            }
        }

    compute_cortical_depth_without_clustering_ch = downstream_zarrs_ch
        .join(compute_cortical_depth_without_clustering_gate_ch)
        .map { key, pairId, platform, latestZarr, row, analysisSegmentations ->
            tuple(key, pairId, platform, latestZarr, row, analysisSegmentations)
        }

    compute_cortical_depth_inputs_ch = compute_cortical_depth_after_clustering_ch
        .mix(compute_cortical_depth_without_clustering_ch)
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

    distance_from_object_after_cortical_gate_ch =
        post_pciseq_sample_rows_ch.flatMap { pairId, row, settings ->
            if (!(settings.run_distance_from_object &&
                  settings.run_compute_cortical_depth)) {
                []
            } else {
                settings.active_platforms.collect { platform ->
                    tuple(
                        "${pairId}|${platform}",
                        row,
                        settings.distance_from_object_segmentations,
                    )
                }
            }
        }

    distance_from_object_after_cortical_ch = compute_cortical_depth_results_ch
        .join(distance_from_object_after_cortical_gate_ch)
        .map {
            key,
            pairId,
            platform,
            latestZarr,
            _corticalDepthOut,
            row,
            distanceSegmentations ->
                tuple(
                    key,
                    pairId,
                    platform,
                    latestZarr,
                    row,
                    distanceSegmentations,
                )
        }

    distance_from_object_after_clustering_gate_ch =
        post_pciseq_sample_rows_ch.flatMap { pairId, row, settings ->
            if (!(settings.run_distance_from_object &&
                  !settings.run_compute_cortical_depth &&
                  settings.run_clustering_squidpy)) {
                []
            } else {
                settings.active_platforms.collect { platform ->
                    tuple(
                        "${pairId}|${platform}",
                        row,
                        settings.distance_from_object_segmentations,
                    )
                }
            }
        }

    distance_from_object_after_clustering_ch = downstream_zarrs_ch
        .join(distance_from_object_after_clustering_gate_ch)
        .map {
            key, pairId, platform, latestZarr, row, distanceSegmentations ->
                tuple(
                    pairId,
                    key,
                    platform,
                    latestZarr,
                    row,
                    distanceSegmentations,
                )
        }
        .combine(clustering_done_per_pair_ch, by: 0)
        .map {
            pairId,
            key,
            platform,
            latestZarr,
            row,
            distanceSegmentations,
            _done ->
                tuple(
                    key,
                    pairId,
                    platform,
                    latestZarr,
                    row,
                    distanceSegmentations,
                )
        }

    distance_from_object_without_terminal_dependency_gate_ch =
        post_pciseq_sample_rows_ch.flatMap { pairId, row, settings ->
            if (!(settings.run_distance_from_object &&
                  !settings.run_compute_cortical_depth &&
                  !settings.run_clustering_squidpy)) {
                []
            } else {
                settings.active_platforms.collect { platform ->
                    tuple(
                        "${pairId}|${platform}",
                        row,
                        settings.distance_from_object_segmentations,
                    )
                }
            }
        }

    distance_from_object_without_terminal_dependency_ch = downstream_zarrs_ch
        .join(distance_from_object_without_terminal_dependency_gate_ch)
        .map {
            key, pairId, platform, latestZarr, row, distanceSegmentations ->
                tuple(
                    key,
                    pairId,
                    platform,
                    latestZarr,
                    row,
                    distanceSegmentations,
                )
        }

    distance_from_object_inputs_ch = distance_from_object_after_cortical_ch
        .mix(distance_from_object_after_clustering_ch)
        .mix(distance_from_object_without_terminal_dependency_ch)
        .map {
            key, pairId, platform, latestZarr, row, distanceSegmentations ->
                def distanceConfig = distanceFromObjectConfigForPlatform(
                    pairId,
                    platform,
                    distanceSegmentations,
                    params,
                )
                def annotationPath = file(
                    normalizedPath(distanceFromObjectAnnotationPath(row, platform))
                )
                tuple(
                    key,
                    pairId,
                    platform,
                    distanceSegmentations,
                    groovy.json.JsonOutput.prettyPrint(
                        groovy.json.JsonOutput.toJson(distanceConfig)
                    ),
                    latestZarr,
                    annotationPath,
                )
        }

    distance_from_object_annotation_results_ch = DISTANCE_FROM_OBJECT_ANNOTATE(
        distance_from_object_inputs_ch
    )

    distance_from_object_cohort_inputs_ch = distance_from_object_annotation_results_ch
        .map {
            _key, _pairId, platform, distanceSegmentations,
            _latestZarr, annotationOutputDir ->
                tuple(platform, annotationOutputDir, distanceSegmentations)
        }
        .groupTuple()
        .map { platform, annotationOutputDirs, segmentationLists ->
            def annotatedSegmentations = segmentationLists
                .flatten()
                .collect { value -> value.toString() }
                .unique()
            def cohortConfig = [
                platform: platform,
                annotation_output_dirs: ["pair_outputs"],
                output_dir: "distance_from_object_cohort_out",
                segmentations: annotatedSegmentations,
                min_pairs: params.distance_from_object_min_pairs,
                n_cpus: params.distance_from_object_n_cpus,
            ]
            tuple(
                platform,
                groovy.json.JsonOutput.prettyPrint(
                    groovy.json.JsonOutput.toJson(cohortConfig)
                ),
                annotationOutputDirs,
            )
        }

    DISTANCE_FROM_OBJECT_COHORT(distance_from_object_cohort_inputs_ch)

    mapmycells_after_clustering_gate_ch = post_pciseq_sample_rows_ch.flatMap { pairId, _row, settings ->
        if (!(settings.run_mapmycells && settings.run_clustering_squidpy)) {
            []
        } else {
            validateMapMyCellsParams(params)
            settings.analysis_segmentations.collect { segmentation ->
                tuple("${pairId}|${segmentation}", true)
            }
        }
    }

    mapmycells_from_clustering_ch = clustering_downstream_results_ch
        .map { pairId, segmentation, samplesJson, clusteringOutDir ->
            tuple("${pairId}|${segmentation}", pairId, segmentation, samplesJson, clusteringOutDir)
        }
        .join(mapmycells_after_clustering_gate_ch)
        .map { _branchKey, pairId, segmentation, samplesJson, clusteringOutDir, _runFlag ->
            tuple(pairId, segmentation, samplesJson, clusteringOutDir)
        }

    mapmycells_published_ch = post_pciseq_sample_rows_ch.flatMap { pairId, _row, settings ->
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

    mapmycells_results_ch = MAPMYCELLS(mapmycells_inputs_ch)

    // Reusable per-pair barrier for independent terminal analyses. Each event
    // is grouped with a known pair-specific cardinality, so no pair waits for
    // another pair. Cohort summaries are intentionally outside the barrier.
    // MENDER consumes this token without depending on other terminal analyses.
    pair_terminal_specs_ch = post_pciseq_sample_rows_ch.flatMap { pairId, _row, settings ->
        if (!settings.run_terminal_analyses) {
            []
        } else {
            def terminalStage = currentPairTerminalStage(settings)
            if (terminalStage == null) {
                []
            } else {
                [tuple(
                    pairId,
                    terminalStage,
                    currentPairTerminalExpectedCount(settings, terminalStage),
                )]
            }
        }
    }

    pair_terminal_immediate_ch = post_pciseq_sample_rows_ch.flatMap {
        pairId, _row, settings ->
            if (settings.run_terminal_analyses && currentPairTerminalStage(settings) == null) {
                [tuple(pairId, true)]
            } else {
                []
            }
    }

    mapmycells_terminal_events_ch = mapmycells_results_ch.map {
        pairId, _segmentation, _mapmycellsOut ->
            tuple(pairId, "mapmycells", true)
    }
    distance_terminal_events_ch = distance_from_object_annotation_results_ch.map {
        _key, pairId, _platform, _distanceSegmentations,
        _latestZarr, _annotationOutputDir ->
            tuple(pairId, "distance_from_object_annotation", true)
    }
    cortical_terminal_events_ch = compute_cortical_depth_results_ch.map {
        _key, pairId, _platform, _latestZarr, _corticalDepthOut ->
            tuple(pairId, "compute_cortical_depth", true)
    }
    clustering_terminal_events_ch = clustering_downstream_results_ch.map {
        pairId, _segmentation, _samplesJson, _clusteringOutDir ->
            tuple(pairId, "clustering_squidpy", true)
    }
    terminal_extra_clustering_gate_ch = post_pciseq_sample_rows_ch.flatMap {
        pairId, _row, settings ->
            if (!(settings.run_terminal_analyses &&
                  settings.run_mapmycells &&
                  settings.run_clustering_squidpy)) {
                []
            } else {
                settings.required_clustering_segmentations
                    .findAll { segmentation ->
                        !settings.analysis_segmentations.contains(segmentation)
                    }
                    .collect { segmentation ->
                        tuple("${pairId}|${segmentation}", pairId)
                    }
            }
    }
    terminal_extra_clustering_terminal_events_ch = clustering_downstream_results_ch
        .map { pairId, segmentation, _samplesJson, _clusteringOutDir ->
            tuple("${pairId}|${segmentation}", pairId)
        }
        .join(terminal_extra_clustering_gate_ch)
        .map { _branchKey, pairId, _expectedPairId ->
            tuple(pairId, "mapmycells", true)
        }
    pair_terminal_events_ch = mapmycells_terminal_events_ch
        .mix(distance_terminal_events_ch)
        .mix(cortical_terminal_events_ch)
        .mix(clustering_terminal_events_ch)
        .mix(terminal_extra_clustering_terminal_events_ch)

    // Pair every terminal event with the pair's one spec: combine keeps the
    // spec, whereas join pairs items one to one, so the pair's first terminal
    // event (often FINALIZE's) would consume it and the events the terminal
    // consumers wait for would be dropped. Legacy and map_first runs alike.
    pair_terminal_grouped_ch = pair_terminal_events_ch
        .combine(pair_terminal_specs_ch, by: 0)
        .filter { _pairId, eventStage, _done, expectedStage, _expectedCount ->
            eventStage == expectedStage
        }
        .map { pairId, _eventStage, _done, _expectedStage, expectedCount ->
            tuple(groupKey(pairId, expectedCount as int), true)
        }
        .groupTuple()
        .map { pairKey, _doneFlags -> tuple(pairKey.getGroupTarget(), true) }

    pair_terminal_token_ch = pair_terminal_immediate_ch.mix(
        pair_terminal_grouped_ch
    )

    mender_current_clustering_gate_ch = post_pciseq_sample_rows_ch.flatMap {
        pairId, _row, settings ->
            if (!(settings.run_mender && settings.run_clustering_squidpy)) {
                []
            } else {
                settings.mender_segmentations.collect { segmentation ->
                    tuple("${pairId}|${segmentation}", settings)
                }
            }
    }

    mender_current_artifacts_ch = clustering_downstream_results_ch
        .map { pairId, segmentation, samplesJson, clusteringOutDir ->
            tuple(
                "${pairId}|${segmentation}",
                pairId,
                segmentation,
                samplesJson,
                clusteringOutDir,
            )
        }
        .join(mender_current_clustering_gate_ch)
        .flatMap {
            _segmentationKey,
            pairId,
            segmentation,
            samplesJson,
            clusteringOutDir,
            settings ->
                def samples = new groovy.json.JsonSlurper().parseText(
                    samplesJson.toString()
                )
                samples.collect { sample ->
                    def platform = sample.platform.toString()
                    def sampleId = sample.sample_id.toString()
                    def clusteredH5ad = java.nio.file.Paths.get(
                        clusteringOutDir.toString()
                    ).resolve(platform.toLowerCase()).resolve(
                        "${sampleId}_clustered.h5ad"
                    )
                    tuple(
                        pairId,
                        "${pairId}|${segmentation}|${platform}",
                        segmentation,
                        platform,
                        sampleId,
                        clusteredH5ad,
                        java.nio.file.Paths.get(
                            sample.zarr_path.toString()
                        ).toRealPath().toString(),
                        settings,
                    )
                }
        }

    mender_published_artifacts_ch = post_pciseq_sample_rows_ch.flatMap {
        pairId, _row, settings ->
            if (!(settings.run_mender && settings.mender_published_output_mode)) {
                []
            } else {
                settings.mender_segmentations.collectMany { segmentation ->
                    settings.active_platforms.collect { platform ->
                        def sampleId = "${pairId}_${platform}"
                        def clusteredH5ad = normalizedPath(
                            publishedPairPath(
                                params.outdir,
                                pairId,
                                "${segmentation}/" +
                                AnnotationSettings.publishedClusteringDir(
                                    params.outdir, pairId, segmentation, platform, sampleId,
                                    settings.clustering_squidpy_table_key_suffix, // rca-site:H2
                                ) +
                                "/clustering_squidpy_out/" +
                                "${platform.toLowerCase()}/" +
                                "${sampleId}_clustered.h5ad",
                            )
                        )
                        def latestZarr = normalizedPath(
                            publishedDatasetPath(
                                params.outdir,
                                pairId,
                                platform,
                                "latest/latest_spatialdata.zarr",
                            )
                        ).toRealPath().toString()
                        tuple(
                            pairId,
                            "${pairId}|${segmentation}|${platform}",
                            segmentation,
                            platform,
                            sampleId,
                            clusteredH5ad,
                            latestZarr,
                            settings,
                        )
                    }
                }
            }
    }

    mender_artifacts_ch = mender_current_artifacts_ch.mix(
        mender_published_artifacts_ch
    )

    mender_inputs_ch = mender_artifacts_ch
        .combine(pair_terminal_token_ch, by: 0)
        .map {
            pairId,
            taskKey,
            segmentation,
            platform,
            sampleId,
            clusteredH5ad,
            latestZarr,
            settings, // rca-site:H2: was _settings; binds the row suffix used below
            terminalToken ->
                def layerKeys = analysisLayerKeys(platform, segmentation)
                def clusteredTableKey = clusteredSpatialdataTableKey(
                    layerKeys.table_key,
                    segmentation,
                    settings.clustering_squidpy_table_key_suffix, // rca-site:H2
                )
                tuple(
                    taskKey,
                    pairId,
                    segmentation,
                    platform,
                    sampleId,
                    latestZarr,
                    clusteredTableKey,
                    layerKeys.shape_key,
                    clusteredH5ad.toAbsolutePath().toString(),
                    clusteredH5ad,
                    terminalToken,
                )
        }

    mender_finalize_lookup_ch = mender_inputs_ch.map {
        taskKey,
        _pairId,
        _segmentation,
        _platform,
        _sampleId,
        _latestZarr,
        _clusteredTableKey,
        _nativeShapeKey,
        _sourceH5adPath,
        clusteredH5ad,
        _terminalToken ->
            tuple(taskKey, clusteredH5ad)
    }

    mender_prepared_ch = MENDER_PREPARE(mender_inputs_ch)
    mender_computed_ch = MENDER_COMPUTE(mender_prepared_ch)
    mender_finalize_inputs_ch = mender_computed_ch
        .map {
            taskKey,
            pairId,
            segmentation,
            platform,
            sampleId,
            spatialdataPath,
            sourceSpatialdataTable,
            nativeShapeKey,
            sourceH5adPath,
            menderConfig,
            preparedDir,
            computedDir ->
                tuple(
                    taskKey,
                    pairId,
                    segmentation,
                    platform,
                    sampleId,
                    spatialdataPath,
                    sourceSpatialdataTable,
                    nativeShapeKey,
                    sourceH5adPath,
                    menderConfig,
                    preparedDir,
                    computedDir,
                )
        }
        .join(mender_finalize_lookup_ch)
        .map {
            taskKey,
            pairId,
            segmentation,
            platform,
            sampleId,
            spatialdataPath,
            sourceSpatialdataTable,
            nativeShapeKey,
            sourceH5adPath,
            menderConfig,
            preparedDir,
            computedDir,
            clusteredH5ad ->
                tuple(
                    taskKey,
                    pairId,
                    segmentation,
                    platform,
                    sampleId,
                    spatialdataPath,
                    sourceSpatialdataTable,
                    nativeShapeKey,
                    sourceH5adPath,
                    menderConfig,
                    preparedDir,
                    computedDir,
                    clusteredH5ad,
                )
        }
    mender_finalized_ch = MENDER_FINALIZE(mender_finalize_inputs_ch)
    MENDER_IMPORT(mender_finalized_ch)
    // Hook H6's record of skipped MENDER runs (no assigned state; map_first only).
    if (AnnotationSettings.isMapFirstRun(params)) { // rca-site:H6
        MENDER_IMPORT.out.subscribe { _taskKey, pairId, segmentation, platform, importManifest ->
            AnnotationRunRecord.menderImported(pairId, segmentation, platform, importManifest)
        }
    }

    gaston_current_clustering_gate_ch = post_pciseq_sample_rows_ch.flatMap {
        pairId, _row, settings ->
            if (!(settings.run_gaston && settings.run_clustering_squidpy)) {
                []
            } else {
                settings.gaston_segmentations.collect { segmentation ->
                    tuple("${pairId}|${segmentation}", settings)
                }
            }
    }

    gaston_current_artifacts_ch = clustering_downstream_results_ch
        .map { pairId, segmentation, samplesJson, clusteringOutDir ->
            tuple(
                "${pairId}|${segmentation}",
                pairId,
                segmentation,
                samplesJson,
                clusteringOutDir,
            )
        }
        .join(gaston_current_clustering_gate_ch)
        .flatMap {
            _segmentationKey,
            pairId,
            segmentation,
            samplesJson,
            clusteringOutDir,
            settings ->
                def samples = new groovy.json.JsonSlurper().parseText(
                    samplesJson.toString()
                )
                samples.collect { sample ->
                    def platform = sample.platform.toString()
                    def sampleId = sample.sample_id.toString()
                    def h5adPath = java.nio.file.Paths.get(
                        clusteringOutDir.toString()
                    ).resolve(platform.toLowerCase()).resolve(
                        "${sampleId}_clustered.h5ad"
                    )
                    tuple(
                        pairId,
                        "${pairId}|${platform}|${segmentation}",
                        segmentation,
                        platform,
                        sampleId,
                        h5adPath,
                        java.nio.file.Paths.get(
                            sample.zarr_path.toString()
                        ).toRealPath().toString(),
                        settings,
                    )
                }
        }

    gaston_published_artifacts_ch = post_pciseq_sample_rows_ch.flatMap {
        pairId, _row, settings ->
            if (!(settings.run_gaston && settings.gaston_published_output_mode)) {
                []
            } else {
                settings.gaston_segmentations.collectMany { segmentation ->
                    settings.active_platforms.collect { platform ->
                        def sampleId = "${pairId}_${platform}"
                        def h5adPath = normalizedPath(
                            publishedPairPath(
                                params.outdir,
                                pairId,
                                "${segmentation}/" +
                                AnnotationSettings.publishedClusteringDir(
                                    params.outdir, pairId, segmentation, platform, sampleId,
                                    settings.clustering_squidpy_table_key_suffix, // rca-site:H2
                                ) +
                                "/clustering_squidpy_out/" +
                                "${platform.toLowerCase()}/" +
                                "${sampleId}_clustered.h5ad",
                            )
                        )
                        def latestZarr = normalizedPath(
                            publishedDatasetPath(
                                params.outdir,
                                pairId,
                                platform,
                                "latest/latest_spatialdata.zarr",
                            )
                        ).toRealPath().toString()
                        tuple(
                            pairId,
                            "${pairId}|${platform}|${segmentation}",
                            segmentation,
                            platform,
                            sampleId,
                            h5adPath,
                            latestZarr,
                            settings,
                        )
                    }
                }
            }
    }

    gaston_artifacts_ch = gaston_current_artifacts_ch.mix(
        gaston_published_artifacts_ch
    )

    gaston_postprocess_config_ch = gaston_artifacts_ch.map {
        pairId,
        branchKey,
        segmentation,
        platform,
        _sampleId,
        _clusteredH5ad,
        latestZarr,
        settings ->
            tuple(
                branchKey,
                gastonConfigJson(
                    pairId,
                    segmentation,
                    platform,
                    latestZarr,
                    settings,
                ),
            )
    }

    gaston_inputs_ch = gaston_artifacts_ch
        .combine(pair_terminal_token_ch, by: 0)
        .map {
            pairId,
            branchKey,
            segmentation,
            platform,
            sampleId,
            clusteredH5ad,
            latestZarr,
            settings,
            terminalToken ->
                tuple(
                    branchKey,
                    pairId,
                    segmentation,
                    platform,
                    sampleId,
                    gastonTrainingConfigJson(
                        pairId,
                        segmentation,
                        platform,
                        latestZarr,
                        settings,
                    ),
                    clusteredH5ad,
                    latestZarr,
                    settings.gaston.n_restarts,
                    settings.gaston.use_gpu,
                    terminalToken,
                )
        }

    gaston_import_lookup_ch = gaston_inputs_ch.map {
        branchKey,
        _pairId,
        _segmentation,
        _platform,
        _sampleId,
        _configJson,
        clusteredH5ad,
        latestZarr,
        _nRestarts,
        _useGpu,
        _terminalToken ->
            tuple(branchKey, clusteredH5ad, latestZarr)
    }

    gaston_prepared_ch = GASTON_PREPARE(gaston_inputs_ch)
    gaston_glmpca_ch = GASTON_GLM_PCA(gaston_prepared_ch)

    gaston_postprocess_metadata_ch = gaston_glmpca_ch
        .join(gaston_postprocess_config_ch)
        .map {
            branchKey,
            pairId,
            segmentation,
            platform,
            sampleId,
            _trainingConfig,
            preparedDir,
            glmpcaDir,
            latestZarr,
            _nRestarts,
            _useGpu,
            postprocessConfigJson ->
                tuple(
                    branchKey,
                    pairId,
                    segmentation,
                    platform,
                    sampleId,
                    postprocessConfigJson,
                    preparedDir,
                    glmpcaDir,
                    latestZarr,
                )
        }

    gaston_train_inputs_ch = gaston_glmpca_ch.flatMap {
        branchKey,
        pairId,
        segmentation,
        platform,
        sampleId,
        gastonConfig,
        preparedDir,
        glmpcaDir,
        latestZarr,
        nRestarts,
        useGpu ->
            (0..<nRestarts).collect { seed ->
                tuple(
                    groupKey(branchKey, nRestarts as int),
                    pairId,
                    segmentation,
                    platform,
                    sampleId,
                    gastonConfig,
                    preparedDir,
                    glmpcaDir,
                    latestZarr,
                    seed,
                    useGpu,
                )
            }
    }

    gaston_seed_results_ch = GASTON_TRAIN(gaston_train_inputs_ch)
    gaston_seed_groups_ch = gaston_seed_results_ch
        .groupTuple()
        .map { branchGroupKey, seedDirs ->
            tuple(branchGroupKey.getGroupTarget(), seedDirs)
        }

    gaston_postprocess_inputs_ch = gaston_seed_groups_ch
        .join(gaston_postprocess_metadata_ch)
        .map {
            branchKey,
            seedDirs,
            pairId,
            segmentation,
            platform,
            sampleId,
            postprocessConfigJson,
            preparedDir,
            glmpcaDir,
            latestZarr ->
                tuple(
                    branchKey,
                    pairId,
                    segmentation,
                    platform,
                    sampleId,
                    postprocessConfigJson,
                    preparedDir,
                    glmpcaDir,
                    latestZarr,
                    seedDirs,
                )
        }

    gaston_standalone_ch = GASTON_POSTPROCESS(gaston_postprocess_inputs_ch)
    gaston_import_inputs_ch = gaston_standalone_ch
        .join(gaston_import_lookup_ch)
        .map {
            branchKey,
            pairId,
            segmentation,
            platform,
            sampleId,
            gastonConfig,
            standaloneDir,
            _postprocessLatestZarr,
            clusteredH5ad,
            latestZarr ->
                tuple(
                    branchKey,
                    pairId,
                    segmentation,
                    platform,
                    sampleId,
                    gastonConfig,
                    standaloneDir,
                    clusteredH5ad,
                    latestZarr,
                )
        }

    GASTON_IMPORT(gaston_import_inputs_ch)

    if (AnnotationSettings.isMapFirstRun(params) && params.annotation_hippocampal_resources && AnnotationReport.enabled(params)) {
        hippocampal_labels_ch.subscribe { pairId, segmentation, _resolved ->
            AnnotationRunRecord.reportExpected(pairId, segmentation)
        }
        hippocampal_report_rows = hippocampal_labels_ch
            .map { pairId, segmentation, resolved -> tuple(AnnotationReferences.branchKey(pairId, segmentation), pairId, segmentation, resolved) }
            .join(clustering_results_ch.map { pairId, segmentation, _samples, clustered -> tuple(AnnotationReferences.branchKey(pairId, segmentation), clustered) })
            .map { key, pairId, segmentation, resolved, clustered -> tuple(pairId, key, segmentation, resolved, clustered) }
            .combine(sample_rows_ch.map { pairId, _row, settings -> tuple(pairId, settings.run_mender ? settings.mender_segmentations : []) }, by: 0)
            .map { pairId, key, segmentation, resolved, clustered, menderSegmentations -> tuple(key, pairId, segmentation, resolved, clustered, menderSegmentations) }
        hippocampal_mender = mender_finalized_ch
            .filter { _taskKey, _pairId, _segmentation, platform, _sampleId, _spatialdataPath, _sourceTable, _shape, _sourceH5ad, _config, _menderDir -> platform == "ATERA" }
            .map { _taskKey, pairId, segmentation, _platform, _sampleId, _spatialdataPath, _sourceTable, _shape, _sourceH5ad, _config, menderDir -> tuple(AnnotationReferences.branchKey(pairId, segmentation), menderDir) }
        hippocampal_reports_ch = HIPPOCAMPAL_REPORT(hippocampal_report_rows.join(hippocampal_mender, remainder: true)
            .filter { _key, pairId, _segmentation, _resolved, _clustered, _menderSegmentations, _menderDir -> pairId != null }
            .map { _key, pairId, segmentation, resolved, clustered, menderSegmentations, menderDir ->
                tuple(pairId, segmentation, resolved, clustered, menderDir ? [menderDir] : [],
                    segmentation in menderSegmentations,
                    AnnotationReferences.sourcesFingerprint("${projectDir}/../src", ["merxen/annotation/hippocampal_pipeline.py", "merxen/annotation/hippocampal_report.py"]))
            })
        hippocampal_reports_ch.subscribe { pairId, segmentation, reportDir ->
            AnnotationRunRecord.reported(pairId, segmentation, reportDir)
        }
    }

    // rca-hook:H11: the annotation QC report (plan §3.6, §9; M7), map_first runs only:
    // after FINALIZE, the pair's cortical depth and the branch's MENDER when they run.
    if (AnnotationSettings.isMapFirstRun(params) && AnnotationReport.enabled(params)) {
        ANNOTATION_REPORTING(
            CLUSTERING_MAP_FIRST.out.labels,
            CLUSTERING_MAP_FIRST.out.alignment,
            clustering_results_ch,
            compute_cortical_depth_results_ch.map { _key, pairId, platform, _latestZarr, corticalDepthOut ->
                tuple(pairId, platform, corticalDepthOut)
            },
            mender_finalized_ch.map {
                _taskKey, pairId, segmentation, platform, _sampleId, _spatialdataPath,
                _sourceSpatialdataTable, _nativeShapeKey, _sourceH5adPath, _menderConfig, menderDir ->
                    tuple(pairId, segmentation, platform, menderDir)
            },
            sample_rows_ch.map { pairId, _row, settings -> tuple(pairId, AnnotationReport.pairSpec(settings, params)) },
        )
    }

}
