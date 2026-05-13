nextflow.enable.dsl = 2

import groovy.json.JsonOutput
import java.nio.file.Paths

include { BUILD_SPATIALDATA } from "./modules/spatialdata_build"
include { SEGMENT } from "./modules/segmentation"
include { ENRICH } from "./modules/enrichment"
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
            "build_spatialdata, segment, enrich, qc, align, align_qc, " +
            "compare, visualize, clustering_squidpy, mapmycells"
        )
    }
    return aliases[key]
}

def activeStageOrder(alignmentEnabled) {
    def stages = ["build_spatialdata", "segment", "enrich", "qc"]
    if (alignmentEnabled) {
        stages += ["align", "align_qc"]
    }
    stages += ["compare", "visualize", "clustering_squidpy", "mapmycells"]
    return stages
}

def validateStage(stage, stages, paramName, alignmentEnabled) {
    if (!stages.contains(stage)) {
        def hint = stage in ["align", "align_qc"] && !alignmentEnabled
            ? " Pass --enable_alignment true to use alignment stages."
            : ""
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
    def p = Paths.get(rawPath.toString()).toAbsolutePath().normalize()
    if (!p.toFile().exists()) {
        throw new IllegalArgumentException(
            "Missing expected ${label}: ${p}\n" +
            "This path is required because an upstream stage was skipped. " +
            "Either run from an earlier stage or restore the published output."
        )
    }
    return p
}

def publishedDatasetPath(outdir, pairId, platform, suffix) {
    return "${outdir}/${pairId}/${platform.toLowerCase()}/${suffix}"
}

def publishedPairPath(outdir, pairId, suffix) {
    return "${outdir}/${pairId}/${suffix}"
}

workflow {
    if (!params.samplesheet) {
        error "Missing required parameter: --samplesheet"
    }

    def alignmentEnabled = params.enable_alignment.toString().toBoolean()
    def stageOrder = activeStageOrder(alignmentEnabled)
    def onlyStageRaw = params.only_stage == null ? "" : params.only_stage.toString().trim()
    def startStage = normalizeStage(
        onlyStageRaw ? onlyStageRaw : params.start_stage,
        onlyStageRaw ? "only_stage" : "start_stage",
    )
    def stopStage = normalizeStage(
        onlyStageRaw ? onlyStageRaw : params.stop_stage,
        onlyStageRaw ? "only_stage" : "stop_stage",
    )
    validateStage(startStage, stageOrder, "start_stage", alignmentEnabled)
    validateStage(stopStage, stageOrder, "stop_stage", alignmentEnabled)
    if (stageOrder.indexOf(startStage) > stageOrder.indexOf(stopStage)) {
        error "start_stage '${startStage}' comes after stop_stage '${stopStage}'."
    }

    def runBuild = stageInRange("build_spatialdata", startStage, stopStage, stageOrder)
    def runSegment = stageInRange("segment", startStage, stopStage, stageOrder)
    def runEnrich = stageInRange("enrich", startStage, stopStage, stageOrder)
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

    if (runSegment && !params.proseg_binary) {
        error "Missing required parameter for SEGMENT: --proseg_binary"
    }
    if (runMapMyCells) {
        def mapMyCellsReferenceMode = params.mapmycells_reference_mode == null
            ? "both"
            : params.mapmycells_reference_mode.toString().trim().toLowerCase()
        if (!(mapMyCellsReferenceMode in ["whole_brain", "region", "both"])) {
            error(
                "Invalid MAPMYCELLS --mapmycells_reference_mode " +
                "'${params.mapmycells_reference_mode}'. Valid values: " +
                "whole_brain, region, both"
            )
        }
        if (mapMyCellsReferenceMode in ["whole_brain", "both"] &&
            !params.mapmycells_marker_lookup_path) {
            error "Missing required parameter for MAPMYCELLS: --mapmycells_marker_lookup_path"
        }
        if (mapMyCellsReferenceMode in ["whole_brain", "both"] &&
            !params.mapmycells_precomputed_stats_path) {
            error "Missing required parameter for MAPMYCELLS: --mapmycells_precomputed_stats_path"
        }
        if (mapMyCellsReferenceMode in ["region", "both"] &&
            !params.mapmycells_region_labels) {
            error "Missing required parameter for MAPMYCELLS region mode: --mapmycells_region_labels"
        }
    }

    def selectedStages = stageOrder.findAll { stageInRange(it, startStage, stopStage, stageOrder) }
    log.info "Selected stages: ${selectedStages.join(' -> ')}"

    samplesheet_ch = Channel
        .fromPath(params.samplesheet, checkIfExists: true)
        .splitCsv(header: true, sep: ",", quote: '"', strip: true)

    if (runBuild) {
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
                error(
                    "Samplesheet row for ${pairId} must provide " +
                    "merscope_dir or merscope_spatialdata_path"
                )
            }
            if (!xeniumDir && !xeniumSpatialdataPath) {
                error(
                    "Samplesheet row for ${pairId} must provide " +
                    "xenium_dir or xenium_spatialdata_path"
                )
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
    }

    def needBuildResults = runSegment || runEnrich
    if (needBuildResults && !runBuild) {
        build_results_ch = samplesheet_ch.flatMap { row ->
            def pairId = row.pair_id?.toString()?.trim()
            if (!pairId) {
                error "Found samplesheet row with missing pair_id: ${row}"
            }

            ["MERSCOPE", "XENIUM"].collect { platform ->
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

    if (runSegment) {
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
                        persistent_latest_zarr_path: persistentLatestZarrPath,
                        persistent_mask_path: persistentMaskPath,
                        persistent_transcripts_path: persistentTranscriptsPath,
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
    }

    if (runEnrich && !runSegment) {
        segment_results_ch = samplesheet_ch.flatMap { row ->
            def pairId = row.pair_id?.toString()?.trim()
            if (!pairId) {
                error "Found samplesheet row with missing pair_id: ${row}"
            }

            ["MERSCOPE", "XENIUM"].collect { platform ->
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

    if (runEnrich) {
        metadata_ch = build_results_ch.map { key, pairId, platform, sourceSpatialdata ->
            tuple(key, pairId, platform, sourceSpatialdata.toString())
        }

        enrich_inputs_ch = segment_results_ch
            .join(metadata_ch)
            .map {
                key, pairId, platform, latestZarr, maskPath, transcriptsCsv,
                pairMeta, platformMeta, originalDataPath ->
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
                    JsonOutput.prettyPrint(JsonOutput.toJson(enrichConfig)),
                    latestZarr,
                    maskPath,
                )
            }

        enrich_results_ch = ENRICH(enrich_inputs_ch)
    }

    if (runQc) {
        if (!runEnrich) {
            enrich_results_ch = samplesheet_ch.flatMap { row ->
                def pairId = row.pair_id?.toString()?.trim()
                if (!pairId) {
                    error "Found samplesheet row with missing pair_id: ${row}"
                }

                ["MERSCOPE", "XENIUM"].collect { platform ->
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

        qc_inputs_ch = enrich_results_ch.map {
            key, pairId, platform, enrichedLatestZarr, enrichOutDir ->
                tuple(key, pairId, platform, enrichedLatestZarr)
            }

        qc_results_ch = QC(qc_inputs_ch)
    }

    def needPairedZarrs =
        runAlign || runAlignQc || runCompare || runVisualize || runClusteringSquidpy
    if (needPairedZarrs) {
        if (runQc) {
            merscope_qc_ch = qc_results_ch
                .filter {
                    key, pairId, platform, enrichedLatestZarr, qcOutDir ->
                        platform == "MERSCOPE"
                }
                .map {
                    key, pairId, platform, enrichedLatestZarr, qcOutDir ->
                        tuple(pairId, enrichedLatestZarr)
                }

            xenium_qc_ch = qc_results_ch
                .filter {
                    key, pairId, platform, enrichedLatestZarr, qcOutDir ->
                        platform == "XENIUM"
                }
                .map {
                    key, pairId, platform, enrichedLatestZarr, qcOutDir ->
                        tuple(pairId, enrichedLatestZarr)
                }

            paired_zarrs_ch = merscope_qc_ch
                .join(xenium_qc_ch)
                .map { pairId, merscopeLatest, xeniumLatest ->
                    def merscopePath = Paths.get(merscopeLatest.toString()).toRealPath().toString()
                    def xeniumPath = Paths.get(xeniumLatest.toString()).toRealPath().toString()
                    tuple(
                        pairId,
                        file(merscopeLatest),
                        file(xeniumLatest),
                        merscopePath,
                        xeniumPath,
                    )
                }
        } else {
            paired_zarrs_ch = samplesheet_ch.map { row ->
                def pairId = row.pair_id?.toString()?.trim()
                if (!pairId) {
                    error "Found samplesheet row with missing pair_id: ${row}"
                }
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
                tuple(
                    pairId,
                    merscopeLatest,
                    xeniumLatest,
                    merscopeLatest.toRealPath().toString(),
                    xeniumLatest.toRealPath().toString(),
                )
            }
        }
    }

    if (alignmentEnabled) {
        def needAlignmentResults =
            runAlign || runAlignQc || runCompare || runVisualize || runClusteringSquidpy
        if (needAlignmentResults) {
            if (runAlign) {
                alignment_results_ch = ALIGN(paired_zarrs_ch)
            } else {
                alignment_results_ch = samplesheet_ch.map { row ->
                    def pairId = row.pair_id?.toString()?.trim()
                    if (!pairId) {
                        error "Found samplesheet row with missing pair_id: ${row}"
                    }
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
                    tuple(
                        pairId,
                        merscopeLatest.toRealPath().toString(),
                        xeniumLatest.toRealPath().toString(),
                        transformJson,
                        coordsDir,
                    )
                }
            }

            if (runAlignQc) {
                ALIGN_QC(alignment_results_ch)
            }

            downstream_zarrs_ch = alignment_results_ch.map {
                pairId, merscopeLatest, xeniumLatest, transformJson, coordsDir ->
                    tuple(pairId, file(merscopeLatest), xeniumLatest)
            }
        }
    } else if (runCompare || runVisualize || runClusteringSquidpy) {
        downstream_zarrs_ch = paired_zarrs_ch.map {
            pairId, merscopeLatest, xeniumLatest, merscopePath, xeniumPath ->
                tuple(pairId, merscopeLatest, xeniumPath)
        }
    }

    if (runCompare) {
        compare_results_ch = COMPARE(downstream_zarrs_ch)
    }

    if (runVisualize) {
        if (runCompare) {
            compare_done_ch = compare_results_ch.map { pairId, compareOutDir ->
                tuple(pairId, true)
            }

            visualize_inputs_ch = downstream_zarrs_ch
                .join(compare_done_ch)
                .map { pairId, merscopeLatest, xeniumLatest, doneFlag ->
                    tuple(pairId, merscopeLatest, xeniumLatest)
                }
        } else {
            visualize_inputs_ch = downstream_zarrs_ch
        }

        visualize_results_ch = VISUALIZE(visualize_inputs_ch)
    }

    if (runClusteringSquidpy) {
        if (runVisualize) {
            visualize_done_ch = visualize_results_ch.map { pairId, visualizeOutDir ->
                tuple(pairId, true)
            }

            clustering_inputs_ch = downstream_zarrs_ch
                .join(visualize_done_ch)
                .map { pairId, merscopeLatest, xeniumLatest, doneFlag ->
                    tuple(pairId, merscopeLatest, xeniumLatest)
                }
        } else {
            clustering_inputs_ch = downstream_zarrs_ch
        }

        clustering_results_ch = CLUSTERING_SQUIDPY(clustering_inputs_ch)
    }

    if (runMapMyCells) {
        if (!runClusteringSquidpy) {
            clustering_results_ch = samplesheet_ch.map { row ->
                def pairId = row.pair_id?.toString()?.trim()
                if (!pairId) {
                    error "Found samplesheet row with missing pair_id: ${row}"
                }
                def clusteringOut = requireExistingPath(
                    publishedPairPath(
                        params.outdir,
                        pairId,
                        "clustering_squidpy/clustering_squidpy_out",
                    ),
                    "CLUSTERING_SQUIDPY output directory for ${pairId}",
                )
                tuple(pairId, clusteringOut)
            }
        }

        MAPMYCELLS(clustering_results_ch)
    }
}
