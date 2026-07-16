process CELLPOSE_SEGMENT {
    tag "${pair_id}:${platform}"

    publishDir { "${params.outdir}/${pair_id}/${platform.toLowerCase()}/segmentation" }, mode: "symlink", overwrite: true

    input:
    tuple val(key), val(pair_id), val(platform), val(seg_config_json)

    output:
    tuple val(key), val(pair_id), val(platform), val(seg_config_json), path("segment_out/cellpose_masks_tiled.npy"), path("segment_out/transcripts_for_proseg.csv"), path("segment_out/cellpose_transforms.json"), path("segment_out/cellpose_stitching_stats.json")

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


process CELLPOSE_NUCLEI_SEGMENT {
    tag "${pair_id}:${platform}:nuclei"

    publishDir { "${params.outdir}/${pair_id}/${platform.toLowerCase()}/segmentation" }, mode: "symlink", overwrite: true

    input:
    tuple val(key), val(pair_id), val(platform), val(seg_config_json)

    output:
    tuple val(key), val(pair_id), val(platform), val(seg_config_json), path("segment_out/cellpose_nuclei_masks_tiled.npy"), path("segment_out/cellpose_nuclei_stitching_stats.json")

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

    merxen cellpose-nuclei-segment --config segment_config.json
    """
}


process PROSEG_SEGMENT {
    tag "${pair_id}:${platform}"

    publishDir { "${params.outdir}/${pair_id}/${platform.toLowerCase()}/segmentation" }, mode: "symlink", overwrite: true

    input:
    tuple val(key), val(pair_id), val(platform), val(seg_config_json), path(cellpose_mask), path(transcripts_csv), path(cellpose_transforms), path(_stitching_stats), path(nuclei_mask), path(_nuclei_stitching_stats), path(proseg_path_file)

    output:
    tuple val(key), val(pair_id), val(platform), path("segment_out/proseg_base_latest.zarr"), path(cellpose_mask), path(transcripts_csv), path(nuclei_mask)

    script:
    """
    set -euo pipefail
    export OMP_NUM_THREADS="${task.cpus}"
    export OPENBLAS_NUM_THREADS="${task.cpus}"
    export MKL_NUM_THREADS="${task.cpus}"
    export NUMEXPR_NUM_THREADS="${task.cpus}"
    export NUMBA_NUM_THREADS="${task.cpus}"
    export VECLIB_MAXIMUM_THREADS="${task.cpus}"
    export BLIS_NUM_THREADS="${task.cpus}"
    export RAYON_NUM_THREADS="${task.cpus}"
    export POLARS_MAX_THREADS="${task.cpus}"
    export DASK_NUM_WORKERS="${task.cpus}"

    cat > segment_config.json <<'JSON'
${seg_config_json}
JSON

    merxen proseg-segment \
        --config segment_config.json \
        --cellpose-mask "${cellpose_mask}" \
        --transcripts-csv "${transcripts_csv}" \
        --cellpose-transforms "${cellpose_transforms}" \
        --proseg-binary "\$(cat "${proseg_path_file}")"
    """
}


workflow SEGMENT {
    take:
    segment_inputs
    nuclei_results
    proseg_path

    main:
    cellpose_results = CELLPOSE_SEGMENT(segment_inputs)
    combined_cellpose_results = cellpose_results
        .join(nuclei_results)
        .map {
            key, pair_id, platform, seg_config_json, cellpose_mask,
            transcripts_csv, cellpose_transforms, stitching_stats,
            nuclei_pair_id, nuclei_platform, _nuclei_seg_config_json,
            nuclei_mask, nuclei_stitching_stats ->
            if (pair_id != nuclei_pair_id || platform != nuclei_platform) {
                error("Cell/nuclei Cellpose channel mismatch for ${key}")
            }
            tuple(
                key,
                pair_id,
                platform,
                seg_config_json,
                cellpose_mask,
                transcripts_csv,
                cellpose_transforms,
                stitching_stats,
                nuclei_mask,
                nuclei_stitching_stats,
            )
        }
    proseg_inputs = combined_cellpose_results
        .combine(proseg_path)
        .map {
            key, pair_id, platform, seg_config_json, cellpose_mask,
            transcripts_csv, cellpose_transforms, stitching_stats, nuclei_mask,
            nuclei_stitching_stats,
            proseg_path_file ->
            tuple(
                key,
                pair_id,
                platform,
                seg_config_json,
                cellpose_mask,
                transcripts_csv,
                cellpose_transforms,
                stitching_stats,
                nuclei_mask,
                nuclei_stitching_stats,
                proseg_path_file,
            )
        }
    segment_results = PROSEG_SEGMENT(proseg_inputs)

    emit:
    segment_results
}
