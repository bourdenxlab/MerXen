import groovy.json.JsonOutput
import groovy.json.JsonSlurper

process MAPMYCELLS {
    tag "${pair_id}"

    publishDir { "${params.outdir}/${pair_id}/mapmycells" }, mode: "copy", overwrite: true

    input:
    tuple val(pair_id),
        path(clustering_out_dir, stageAs: "clustering_squidpy_input")

    output:
    tuple val(pair_id), path("mapmycells_out")

    script:
    def dropLevelJson = params.mapmycells_drop_level == null ? "null" : JsonOutput.toJson(params.mapmycells_drop_level.toString())
    def queryLayerJson = params.mapmycells_query_layer == null ? "null" : JsonOutput.toJson(params.mapmycells_query_layer.toString())
    def geneIdColumnJson = params.mapmycells_gene_id_column == null ? "null" : JsonOutput.toJson(params.mapmycells_gene_id_column.toString())
    def obsIdColumnJson = params.mapmycells_obs_id_column == null ? "null" : JsonOutput.toJson(params.mapmycells_obs_id_column.toString())
    def tmpDirJson = params.mapmycells_tmp_dir == null ? "null" : JsonOutput.toJson(params.mapmycells_tmp_dir.toString())
    def referenceModeJson = params.mapmycells_reference_mode == null ? JsonOutput.toJson("both") : JsonOutput.toJson(params.mapmycells_reference_mode.toString())
    def markerLookupJson = params.mapmycells_marker_lookup_path == null ? "null" : JsonOutput.toJson(params.mapmycells_marker_lookup_path.toString())
    def precomputedStatsJson = params.mapmycells_precomputed_stats_path == null ? "null" : JsonOutput.toJson(params.mapmycells_precomputed_stats_path.toString())
    def regionNameJson = params.mapmycells_region_name == null ? JsonOutput.toJson("region") : JsonOutput.toJson(params.mapmycells_region_name.toString())
    def regionCacheDirJson = params.mapmycells_region_cache_dir == null ? "null" : JsonOutput.toJson(params.mapmycells_region_cache_dir.toString())
    def regionLabelValues = []
    if (params.mapmycells_region_labels instanceof List) {
        regionLabelValues = params.mapmycells_region_labels.collect { it.toString() }
    } else if (params.mapmycells_region_labels != null) {
        def rawRegionLabels = params.mapmycells_region_labels.toString().trim()
        if (rawRegionLabels.startsWith("[")) {
            regionLabelValues = new JsonSlurper().parseText(rawRegionLabels).collect { it.toString() }
        } else if (rawRegionLabels) {
            regionLabelValues = rawRegionLabels
                .split(",")
                .collect { it.trim() }
                .findAll { it.length() > 0 }
        }
    }
    def regionLabelsJson = JsonOutput.toJson(regionLabelValues)
    """
    set -euo pipefail

    cat > mapmycells_config.json <<JSON
{
  "pair_id": "${pair_id}",
  "output_dir": "mapmycells_out",
  "samples": [
    {
      "sample_id": "${pair_id}_MERSCOPE",
      "platform": "MERSCOPE",
      "anndata_path": "${clustering_out_dir}/merscope/${pair_id}_MERSCOPE_clustered.h5ad",
      "query_layer": ${queryLayerJson},
      "gene_id_column": ${geneIdColumnJson},
      "obs_id_column": ${obsIdColumnJson}
    },
    {
      "sample_id": "${pair_id}_XENIUM",
      "platform": "XENIUM",
      "anndata_path": "${clustering_out_dir}/xenium/${pair_id}_XENIUM_clustered.h5ad",
      "query_layer": ${queryLayerJson},
      "gene_id_column": ${geneIdColumnJson},
      "obs_id_column": ${obsIdColumnJson}
    }
  ],
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
  "verbose_csv": ${params.mapmycells_verbose_csv}
}
JSON

    merxen mapmycells --config mapmycells_config.json
    """
}
