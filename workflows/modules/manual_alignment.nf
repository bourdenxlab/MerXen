process MANUAL_ALIGN {
    tag "${pair_id}"

    publishDir { "${params.outdir}/${pair_id}/alignment" }, mode: "copy", overwrite: true

    input:
    tuple val(pair_id), val(merscope_zarr_path), val(xenium_zarr_path),
        path(prepared_store), path(landmarks, stageAs: "landmarks.json")

    output:
    tuple val(pair_id), val(merscope_zarr_path), val(xenium_zarr_path), path("align_out")

    script:
    def config = [
        pair_id: pair_id,
        merscope_zarr_path: merscope_zarr_path,
        xenium_zarr_path: xenium_zarr_path,
        output_dir: "align_out",
        backend: "manual_landmarks",
        manual: [
            prepared_store: prepared_store.toString(),
            labelling_path: landmarks.toString(),
            settings: params.alignment_manual_settings,
            grid_spacing_um: params.alignment_manual_grid_spacing_um,
        ],
        merscope_image: [
            image_key: params.alignment_merscope_image_key,
            dapi_channel: params.alignment_merscope_dapi_channel,
            pixel_size_um: params.alignment_merscope_pixel_size_um,
        ],
        xenium_image: [
            image_key: params.alignment_xenium_image_key,
            dapi_channel: params.alignment_xenium_dapi_channel,
            pixel_size_um: params.alignment_xenium_pixel_size_um,
        ],
        valis: [coordinate_system_name: params.alignment_coordinate_system_name],
    ]
    def configJson = groovy.json.JsonOutput.prettyPrint(groovy.json.JsonOutput.toJson(config))
    """
    set -euo pipefail
    export PYTHONPATH="${projectDir}/../src:\${PYTHONPATH:-}"
    export OMP_NUM_THREADS="${task.cpus}"
    export OPENBLAS_NUM_THREADS="${task.cpus}"
    export MKL_NUM_THREADS="${task.cpus}"
    cat > align_config.json <<'JSON'
${configJson}
JSON
    merxen align --config align_config.json
    """
}
