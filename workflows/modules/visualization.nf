process VISUALIZE {
    tag "${pair_id}:${segmentation}"

    publishDir { "${params.outdir}/${pair_id}/${segmentation}/visualization" }, mode: "copy", overwrite: true

    input:
    tuple val(pair_id),
        val(segmentation),
        val(samples_json)

    output:
    tuple val(pair_id), val(segmentation), path("visualize_out")

    script:
    """
    set -euo pipefail

    cat > visualize_config.json <<JSON
{
  "pair_id": "${pair_id}",
  "output_dir": "visualize_out",
  "samples": ${samples_json}
}
JSON

    merxen visualize --config visualize_config.json
    """
}
