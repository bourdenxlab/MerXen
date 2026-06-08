process QC {
    tag "${pair_id}:${platform}:${segmentation}"

    publishDir { "${params.outdir}/${pair_id}/${platform.toLowerCase()}/${segmentation}/qc" }, mode: "symlink", overwrite: true

    input:
    tuple val(key),
        val(pair_id),
        val(platform),
        val(segmentation),
        path(latest_zarr),
        val(table_key),
        val(shape_key)

    output:
    tuple val(key),
        val(pair_id),
        val(platform),
        val(segmentation),
        path(latest_zarr),
        path("qc_out"),
        val(table_key),
        val(shape_key)

    script:
    """
    set -euo pipefail

    cat > qc_config.json <<JSON
{
  "dataset_name": "${pair_id}_${platform}",
  "latest_zarr_path": "${latest_zarr}",
  "output_dir": "qc_out",
  "table_key": "${table_key}",
  "shape_key": "${shape_key}"
}
JSON

    merxen qc --config qc_config.json
    """
}
