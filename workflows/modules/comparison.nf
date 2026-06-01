process COMPARE {
    tag "${pair_id}"

    publishDir { "${params.outdir}/${pair_id}/comparison" }, mode: "copy", overwrite: true

    input:
    tuple val(pair_id),
        val(merscope_zarr),
        val(xenium_zarr)

    output:
    tuple val(pair_id), path("compare_out")

    script:
    """
    set -euo pipefail

    cat > compare_config.json <<JSON
{
  "pair_id": "${pair_id}",
  "merscope_zarr_path": "${merscope_zarr}",
  "xenium_zarr_path": "${xenium_zarr}",
  "output_dir": "compare_out"
}
JSON

    merxen compare --config compare_config.json
    """
}
