process MECR_REFERENCE {
    tag "WHB-10Xv3"

    publishDir { "${params.outdir}/mecr_reference" }, mode: "copy", overwrite: true

    input:
    val(samples_json)

    output:
    path("mecr_reference_out")

    script:
    def neuronsPath = groovy.json.JsonOutput.toJson(
        params.mecr_neurons_h5ad_path.toString()
    )
    def nonneuronsPath = groovy.json.JsonOutput.toJson(
        params.mecr_nonneurons_h5ad_path.toString()
    )
    def cellMetadataPath = groovy.json.JsonOutput.toJson(
        params.mecr_cell_metadata_path.toString()
    )
    def taxonomyMetadataPath = groovy.json.JsonOutput.toJson(
        params.mecr_taxonomy_metadata_path.toString()
    )
    def clusterMembershipPath = groovy.json.JsonOutput.toJson(
        params.mecr_cluster_membership_path.toString()
    )
    def targetClasses = groovy.json.JsonOutput.toJson(
        params.mecr_target_broad_classes
    )
    """
    set -euo pipefail
    export OMP_NUM_THREADS="${task.cpus}"
    export OPENBLAS_NUM_THREADS="${task.cpus}"
    export MKL_NUM_THREADS="${task.cpus}"
    export NUMEXPR_NUM_THREADS="${task.cpus}"

    cat > mecr_reference_config.json <<JSON
{
  "output_dir": "mecr_reference_out",
  "samples": ${samples_json},
  "neurons_h5ad_path": ${neuronsPath},
  "nonneurons_h5ad_path": ${nonneuronsPath},
  "cell_metadata_path": ${cellMetadataPath},
  "taxonomy_metadata_path": ${taxonomyMetadataPath},
  "cluster_membership_path": ${clusterMembershipPath},
  "taxonomy_level": "${params.mecr_taxonomy_level}",
  "gene_symbol_column": "${params.mecr_gene_symbol_column}",
  "target_broad_classes": ${targetClasses},
  "marker_min_target_fraction": ${params.mecr_marker_min_target_fraction},
  "marker_max_other_fraction": ${params.mecr_marker_max_other_fraction},
  "normalize_target_sum": ${params.mecr_normalize_target_sum},
  "reference_chunk_rows": ${params.mecr_reference_chunk_rows},
  "wilcoxon_tie_correct": ${params.mecr_wilcoxon_tie_correct},
  "figure_dpi": ${params.mecr_figure_dpi}
}
JSON

    merxen mecr-reference --config mecr_reference_config.json
    """
}


process MECR {
    tag "${pair_id}:${segmentation}"

    publishDir { "${params.outdir}/${pair_id}/${segmentation}/mecr" }, mode: "copy", overwrite: true

    input:
    tuple val(pair_id),
        val(segmentation),
        val(samples_json),
        path(reference_out)

    output:
    tuple val(pair_id),
        val(segmentation),
        path("mecr_out")

    script:
    """
    set -euo pipefail
    export OMP_NUM_THREADS="${task.cpus}"
    export OPENBLAS_NUM_THREADS="${task.cpus}"
    export MKL_NUM_THREADS="${task.cpus}"
    export NUMEXPR_NUM_THREADS="${task.cpus}"

    cat > mecr_config.json <<JSON
{
  "pair_id": "${pair_id}",
  "segmentation": "${segmentation}",
  "output_dir": "mecr_out",
  "samples": ${samples_json},
  "reference_markers_path": "${reference_out}/mecr_reference_markers.csv",
  "figure_dpi": ${params.mecr_figure_dpi},
  "barnyard_top_n_pairs": ${params.mecr_barnyard_top_n_pairs},
  "barnyard_max_points": ${params.mecr_barnyard_max_points},
  "barnyard_random_seed": ${params.mecr_barnyard_random_seed},
  "barnyard_log1p": ${params.mecr_barnyard_log1p}
}
JSON

    merxen mecr --config mecr_config.json
    """
}
