nextflow.enable.dsl = 2

import groovy.json.JsonOutput

include { BUILD_SPATIALDATA } from "./modules/spatialdata_build"
include { SEGMENT } from "./modules/segmentation"
include { ENRICH } from "./modules/enrichment"
include { QC } from "./modules/qc"
include { COMPARE } from "./modules/comparison"
include { VISUALIZE } from "./modules/visualization"

def parseChannels(rawValue, defaults) {
    if (rawValue == null) {
        return defaults
    }
    def values = rawValue
        .toString()
        .split(",")
        .collect { it.trim() }
        .findAll { it.length() > 0 }
    return values ? values : defaults
}

def parseRange(rawValue, fallbackStart = 0, fallbackEnd = 6) {
    if (rawValue == null || rawValue.toString().trim().isEmpty()) {
        return [fallbackStart as int, fallbackEnd as int]
    }
    def parts = rawValue.toString().split("-").collect { it.trim() }
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
    for (name in names) {
        def value = row[name]
        if (value != null && value.toString().trim().length() > 0) {
            return value.toString().trim()
        }
    }
    return null
}

workflow {
    if (!params.samplesheet) {
        error "Missing required parameter: --samplesheet"
    }
    if (!params.proseg_binary) {
        error "Missing required parameter: --proseg_binary"
    }

    samplesheet_ch = Channel
        .fromPath(params.samplesheet, checkIfExists: true)
        .splitCsv(header: true, sep: ",", quote: '"', strip: true)

    build_inputs_ch = samplesheet_ch.flatMap { row ->
        def pairId = row.pair_id?.toString()?.trim()
        if (!pairId) {
            error "Found samplesheet row with missing pair_id: ${row}"
        }

        def merscopeDir = chooseField(row, ["merscope_dir"])
        def merscopeSpatialdataPath = chooseField(
            row,
            ["merscope_spatialdata_path", "merscope_zarr_path"]
        )
        def xeniumDir = chooseField(row, ["xenium_dir"])
        def xeniumSpatialdataPath = chooseField(row, ["xenium_spatialdata_path"])

        if (!merscopeDir && !merscopeSpatialdataPath) {
            error "Samplesheet row for ${pairId} must provide merscope_dir or merscope_spatialdata_path"
        }
        if (!xeniumDir && !xeniumSpatialdataPath) {
            error "Samplesheet row for ${pairId} must provide xenium_dir or xenium_spatialdata_path"
        }

        def merscopeBuildConfig = [
            dataset_name: "${pairId}_MERSCOPE",
            platform: "MERSCOPE",
            input_path: merscopeDir ?: merscopeSpatialdataPath,
            output_path: "spatialdata_out/source_spatialdata.zarr",
            persistent_output_path: merscopeSpatialdataPath ?: null,
            merscope_transform_path: chooseField(row, ["merscope_transform_path"]) ?: null,
            merscope: [:],
            xenium: [:],
        ]

        def xeniumBuildConfig = [
            dataset_name: "${pairId}_XENIUM",
            platform: "XENIUM",
            input_path: xeniumDir ?: xeniumSpatialdataPath,
            output_path: "spatialdata_out/source_spatialdata.zarr",
            persistent_output_path: xeniumSpatialdataPath ?: null,
            xenium_spec_path: chooseField(row, ["xenium_spec_path"]) ?: null,
            merscope: [:],
            xenium: [:],
        ]

        def merscopeKey = "${pairId}|MERSCOPE"
        def xeniumKey = "${pairId}|XENIUM"

        [
            tuple(
                merscopeKey,
                pairId,
                "MERSCOPE",
                JsonOutput.prettyPrint(JsonOutput.toJson(merscopeBuildConfig)),
            ),
            tuple(
                xeniumKey,
                pairId,
                "XENIUM",
                JsonOutput.prettyPrint(JsonOutput.toJson(xeniumBuildConfig)),
            ),
        ]
    }

    build_results_ch = BUILD_SPATIALDATA(build_inputs_ch)

    segment_meta_ch = samplesheet_ch.flatMap { row ->
        def pairId = row.pair_id?.toString()?.trim()
        if (!pairId) {
            error "Found samplesheet row with missing pair_id: ${row}"
        }

        def merscopeVoxelLayers = intOrDefault(
            row.merscope_voxel_layers,
            params.default_merscope_voxel_layers
        )
        def xeniumVoxelLayers = intOrDefault(
            row.xenium_voxel_layers,
            params.default_xenium_voxel_layers
        )
        def xeniumMinQv = floatOrDefault(row.xenium_min_qv, params.xenium_min_qv)
        def merscopeRange = parseRange(row.merscope_z_range, 0, 6)

        def merscopeKey = "${pairId}|MERSCOPE"
        def xeniumKey = "${pairId}|XENIUM"

        [
            tuple(
                merscopeKey,
                [
                    channels: parseChannels(row.merscope_channels, ["DAPI", "PolyT"]),
                    image_prefix: chooseField(row, ["merscope_image_prefix"]) ?: null,
                    z_range: merscopeRange,
                    transform_path: chooseField(row, ["merscope_transform_path"]) ?: null,
                    xenium_spec_path: null,
                    min_qv: null,
                    voxel_layers: merscopeVoxelLayers,
                ],
            ),
            tuple(
                xeniumKey,
                [
                    channels: parseChannels(row.xenium_channels, ["DAPI", "18S"]),
                    image_prefix: null,
                    z_range: null,
                    transform_path: null,
                    xenium_spec_path: chooseField(row, ["xenium_spec_path"]) ?: null,
                    min_qv: xeniumMinQv,
                    voxel_layers: xeniumVoxelLayers,
                ],
            ),
        ]
    }

    segment_inputs_ch = build_results_ch
        .join(segment_meta_ch)
        .map { key, pairId, platform, sourceSpatialdata, meta ->
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
                JsonOutput.prettyPrint(JsonOutput.toJson(segmentConfig)),
            )
        }

    segment_results_ch = SEGMENT(segment_inputs_ch)

    metadata_ch = build_results_ch.map { key, pairId, platform, sourceSpatialdata ->
        tuple(key, pairId, platform, sourceSpatialdata.toString())
    }

    enrich_inputs_ch = segment_results_ch
        .join(metadata_ch)
        .map { key, pairId, platform, latestZarr, maskPath, transcriptsCsv, pairMeta, platformMeta, originalDataPath ->
            if (pairId != pairMeta || platform != platformMeta) {
                error "Internal channel mismatch for key=${key}: ${pairId}/${platform} vs ${pairMeta}/${platformMeta}"
            }

            def enrichConfig = [
                dataset_name: "${pairId}_${platform}",
                platform: platform,
                latest_zarr_path: "latest_input.zarr",
                mask_path: "enrich_input_mask.npy",
                original_data_path: originalDataPath,
                output_dir: "enrich_out",
                transform_path: null,
            ]

            tuple(
                key,
                pairId,
                platform,
                JsonOutput.prettyPrint(JsonOutput.toJson(enrichConfig)),
                latestZarr,
                maskPath,
            )
        }

    enrich_results_ch = ENRICH(enrich_inputs_ch)

    qc_inputs_ch = enrich_results_ch.map { key, pairId, platform, enrichedLatestZarr, enrichOutDir ->
        tuple(key, pairId, platform, enrichedLatestZarr)
    }

    qc_results_ch = QC(qc_inputs_ch)

    merscope_qc_ch = qc_results_ch
        .filter { key, pairId, platform, enrichedLatestZarr, qcOutDir -> platform == "MERSCOPE" }
        .map { key, pairId, platform, enrichedLatestZarr, qcOutDir -> tuple(pairId, enrichedLatestZarr) }

    xenium_qc_ch = qc_results_ch
        .filter { key, pairId, platform, enrichedLatestZarr, qcOutDir -> platform == "XENIUM" }
        .map { key, pairId, platform, enrichedLatestZarr, qcOutDir -> tuple(pairId, enrichedLatestZarr) }

    paired_zarrs_ch = merscope_qc_ch
        .join(xenium_qc_ch)
        .map { pairId, merscopeLatest, xeniumLatest ->
            tuple(pairId, file(merscopeLatest), file(xeniumLatest))
        }

    compare_results_ch = COMPARE(paired_zarrs_ch)

    compare_done_ch = compare_results_ch.map { pairId, compareOutDir ->
        tuple(pairId, true)
    }

    visualize_inputs_ch = paired_zarrs_ch
        .join(compare_done_ch)
        .map { pairId, merscopeLatest, xeniumLatest, doneFlag ->
            tuple(pairId, merscopeLatest, xeniumLatest)
        }

    VISUALIZE(visualize_inputs_ch)
}
