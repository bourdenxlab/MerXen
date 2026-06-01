process VISUALIZE {
    tag "${pair_id}"

    publishDir { "${params.outdir}/${pair_id}/visualization" }, mode: "copy", overwrite: true

    input:
    tuple val(pair_id),
        val(samples_json)

    output:
    tuple val(pair_id), path("visualize_out")

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
