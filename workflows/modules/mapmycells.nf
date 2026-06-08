process MAPMYCELLS {
    tag "${pair_id}:${segmentation}"

    publishDir { "${params.outdir}/${pair_id}/${segmentation}/mapmycells" }, mode: "copy", overwrite: true

    input:
    tuple val(pair_id),
        val(segmentation),
        val(samples_json),
        path(clustering_out_dir, stageAs: "clustering_squidpy_input")

    output:
    tuple val(pair_id), val(segmentation), path("mapmycells_out")

    script:
    def dropLevelJson = params.mapmycells_drop_level == null ? "null" : groovy.json.JsonOutput.toJson(params.mapmycells_drop_level.toString())
    def queryLayerJson = params.mapmycells_query_layer == null ? "null" : groovy.json.JsonOutput.toJson(params.mapmycells_query_layer.toString())
    def geneIdColumnJson = params.mapmycells_gene_id_column == null ? "null" : groovy.json.JsonOutput.toJson(params.mapmycells_gene_id_column.toString())
    def obsIdColumnJson = params.mapmycells_obs_id_column == null ? "null" : groovy.json.JsonOutput.toJson(params.mapmycells_obs_id_column.toString())
    def tmpDirJson = params.mapmycells_tmp_dir == null ? "null" : groovy.json.JsonOutput.toJson(params.mapmycells_tmp_dir.toString())
    def referenceModeJson = params.mapmycells_reference_mode == null ? groovy.json.JsonOutput.toJson("both") : groovy.json.JsonOutput.toJson(params.mapmycells_reference_mode.toString())
    def markerLookupJson = params.mapmycells_marker_lookup_path == null ? "null" : groovy.json.JsonOutput.toJson(params.mapmycells_marker_lookup_path.toString())
    def precomputedStatsJson = params.mapmycells_precomputed_stats_path == null ? "null" : groovy.json.JsonOutput.toJson(params.mapmycells_precomputed_stats_path.toString())
    def regionNameJson = params.mapmycells_region_name == null ? groovy.json.JsonOutput.toJson("region") : groovy.json.JsonOutput.toJson(params.mapmycells_region_name.toString())
    def regionCacheDirJson = params.mapmycells_region_cache_dir == null ? "null" : groovy.json.JsonOutput.toJson(params.mapmycells_region_cache_dir.toString())
    def plotsOnly = params.mapmycells_plots_only == null ? false : params.mapmycells_plots_only.toString().trim().toLowerCase() == "true"
    def plotsOnlyJson = plotsOnly ? "true" : "false"
    def publishedMapMyCellsOut = "${params.outdir}/${pair_id}/${segmentation}/mapmycells/mapmycells_out"
    def clusteringSamples = new groovy.json.JsonSlurper().parseText(samples_json)
    def mapmycellsSamples = clusteringSamples.collect { sample ->
        def sampleId = sample.sample_id.toString()
        def platform = sample.platform.toString()
        [
            sample_id: sampleId,
            platform: platform,
            anndata_path: "${clustering_out_dir}/${platform.toLowerCase()}/${sampleId}_clustered.h5ad",
            query_layer: queryLayerJson == "null" ? null : params.mapmycells_query_layer.toString(),
            gene_id_column: geneIdColumnJson == "null" ? null : params.mapmycells_gene_id_column.toString(),
            obs_id_column: obsIdColumnJson == "null" ? null : params.mapmycells_obs_id_column.toString(),
        ]
    }
    def mapmycellsSamplesJson = groovy.json.JsonOutput.prettyPrint(groovy.json.JsonOutput.toJson(mapmycellsSamples))
    def regionLabelValues = []
    if (params.mapmycells_region_labels instanceof List) {
        regionLabelValues = params.mapmycells_region_labels.collect { it.toString() }
    } else if (params.mapmycells_region_labels != null) {
        def rawRegionLabels = params.mapmycells_region_labels.toString().trim()
        if (rawRegionLabels.startsWith("[")) {
            regionLabelValues = new groovy.json.JsonSlurper().parseText(rawRegionLabels).collect { it.toString() }
        } else if (rawRegionLabels) {
            regionLabelValues = rawRegionLabels
                .split(",")
                .collect { it.trim() }
                .findAll { it.length() > 0 }
        }
    }
    def regionLabelsJson = groovy.json.JsonOutput.toJson(regionLabelValues)
    """
    set -euo pipefail

    if [[ "${plotsOnlyJson}" == "true" ]]; then
        previous_mapmycells_out="${publishedMapMyCellsOut}"
        if [[ ! -d "\${previous_mapmycells_out}" ]]; then
            echo "Missing existing MapMyCells output directory for plots-only mode: \${previous_mapmycells_out}" >&2
            exit 1
        fi
        rm -rf mapmycells_out
        cp -a "\${previous_mapmycells_out}" mapmycells_out
    fi

    cat > mapmycells_config.json <<JSON
{
  "pair_id": "${pair_id}",
  "output_dir": "mapmycells_out",
  "samples": ${mapmycellsSamplesJson},
  "reference_mode": ${referenceModeJson},
  "marker_lookup_path": ${markerLookupJson},
  "precomputed_stats_path": ${precomputedStatsJson},
  "region_name": ${regionNameJson},
  "region_labels": ${regionLabelsJson},
  "region_cache_dir": ${regionCacheDirJson},
  "region_min_cells_per_leaf": ${params.mapmycells_region_min_cells_per_leaf},
  "region_force_rebuild": ${params.mapmycells_region_force_rebuild},
  "region_query_markers_n_per_utility": ${params.mapmycells_region_query_markers_n_per_utility},
  "drop_level": ${dropLevelJson},
  "normalization": "${params.mapmycells_normalization}",
  "bootstrap_factor": ${params.mapmycells_bootstrap_factor},
  "bootstrap_iteration": ${params.mapmycells_bootstrap_iteration},
  "n_processors": ${params.mapmycells_n_processors},
  "chunk_size": ${params.mapmycells_chunk_size},
  "rng_seed": ${params.mapmycells_rng_seed},
  "max_gb": ${params.mapmycells_max_gb},
  "tmp_dir": ${tmpDirJson},
  "cloud_safe": ${params.mapmycells_cloud_safe},
  "flatten": ${params.mapmycells_flatten},
  "verbose_csv": ${params.mapmycells_verbose_csv},
  "plots_only": ${plotsOnlyJson}
}
JSON

    merxen mapmycells --config mapmycells_config.json
    """
}
