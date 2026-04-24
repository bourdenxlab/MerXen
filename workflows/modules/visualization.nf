process VISUALIZE {
    tag "${pair_id}"

    publishDir { "${params.outdir}/${pair_id}/visualization" }, mode: "copy", overwrite: true

    input:
    tuple val(pair_id),
        path(merscope_zarr, stageAs: "merscope_latest_input.zarr"),
        path(xenium_zarr, stageAs: "xenium_latest_input.zarr")

    output:
    tuple val(pair_id), path("visualize_out")

    script:
    """
    set -euo pipefail

    cat > visualize_config.json <<JSON
{
  "pair_id": "${pair_id}",
  "merscope_zarr_path": "${merscope_zarr}",
  "xenium_zarr_path": "${xenium_zarr}",
  "output_dir": "visualize_out"
}
JSON

    merxen visualize --config visualize_config.json
    """
}
