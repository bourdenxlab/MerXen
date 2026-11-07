// ATERA Cellpose (segmentation_source=cellpose_proseg, plan AT-7).
//
// ATERA_CELLPOSE_SEGMENT runs the tiled Cellpose of the XENIUM route on the
// native morphology image of the built ATERA store, on DAPI + 18S. It writes the
// masks, the cell-probability logits, the mask-to-micron transforms, the
// stitching statistics and the model identity, but no ProSeg transcript CSV:
// the tiled ATERA ProSeg route reads the store's parquet. A separate process
// from CELLPOSE_SEGMENT, so the XENIUM and MERSCOPE process (and its task
// hashes) stay as they are.

process ATERA_CELLPOSE_SEGMENT {
    tag "${pair_id}:${platform}"

    publishDir { "${params.outdir}/${pair_id}/${platform.toLowerCase()}/segmentation" }, mode: "symlink", overwrite: true

    input:
    tuple val(key), val(pair_id), val(platform), val(seg_config_json)

    output:
    tuple val(key), val(pair_id), val(platform), path("segment_out/cellpose_masks_tiled.npy"), path("segment_out/cellpose_cellprobs_tiled.npy"), path("segment_out/cellpose_transforms.json"), path("segment_out/cellpose_stitching_stats.json"), emit: segmentation
    path("segment_out/cellpose_model_identity.json"), emit: model_identity

    script:
    """
    set -euo pipefail
    export OMP_NUM_THREADS="${task.cpus}"
    export OPENBLAS_NUM_THREADS="${task.cpus}"
    export MKL_NUM_THREADS="${task.cpus}"
    export NUMEXPR_NUM_THREADS="${task.cpus}"
    export NUMBA_NUM_THREADS="${task.cpus}"

    cat > segment_config.json <<'JSON'
${seg_config_json}
JSON

    merxen cellpose-segment --config segment_config.json
    """
}
