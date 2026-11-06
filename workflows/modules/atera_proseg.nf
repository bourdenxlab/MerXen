// ATERA ProSeg route (segmentation_source=native_proseg, plan AT-9).
//
// ATERA_PROSEG_PLAN cuts the section into tiles; ATERA_PROSEG_TILE runs ProSeg
// on one tile (the vendor cell mask is the prior) and keeps the cells the tile
// owns; ATERA_PROSEG_STITCH writes the stitched cells into the row's latest
// store as the reseg layer (MOSAIK_proseg, table_MOSAIK_proseg,
// proseg_cell_labels, points assignment/background) and emits the SEGMENT
// tuple shape with NATIVE_LATEST's empty placeholders, so ENRICH and every
// later stage take it as they take NATIVE_LATEST's output.

process ATERA_PROSEG_PLAN {
    tag "${pair_id}:${platform}"

    publishDir { "${params.outdir}/${pair_id}/${platform.toLowerCase()}/segmentation/atera_proseg" }, mode: "copy", overwrite: true

    input:
    tuple val(key), val(pair_id), val(platform), val(plan_config_json), path(source_zarr, stageAs: "proseg_source.zarr")

    output:
    tuple val(key), val(pair_id), val(platform), path("plan_out/tile_plan.json"), emit: plan

    script:
    """
    set -euo pipefail
    export OMP_NUM_THREADS="${task.cpus}"
    export POLARS_MAX_THREADS="${task.cpus}"
    export DASK_NUM_WORKERS="${task.cpus}"

    cat > plan_config.json <<'JSON'
${plan_config_json}
JSON

    merxen atera-proseg-plan --config plan_config.json
    """
}

process ATERA_PROSEG_TILE {
    tag "${pair_id}:${platform}:${tile_id}"

    // Copied, not linked: a re-stitch needs each tile's polygons and claims,
    // so cleaning the work directory must not lose them (2.7 GB on the full
    // public section).
    publishDir { "${params.outdir}/${pair_id}/${platform.toLowerCase()}/segmentation/atera_proseg/tiles" }, mode: "copy", overwrite: true

    input:
    tuple val(key), val(pair_id), val(platform), val(n_tiles), val(tile_id), val(tile_config_json), path(proseg_path_file)

    output:
    tuple val(key), val(n_tiles), val(tile_id), path("tile_${tile_id}"), emit: tile

    script:
    """
    set -euo pipefail
    export OMP_NUM_THREADS="${task.cpus}"
    export POLARS_MAX_THREADS="${task.cpus}"
    export DASK_NUM_WORKERS="${task.cpus}"
    export RAYON_NUM_THREADS="${task.cpus}"

    cat > tile_config.json <<'JSON'
${tile_config_json}
JSON

    merxen atera-proseg-tile --config tile_config.json --proseg-binary "\$(cat ${proseg_path_file})"
    """
}

process ATERA_PROSEG_STITCH {
    tag "${pair_id}:${platform}"

    // The store link and the placeholders are linked like NATIVE_LATEST's;
    // the report is copied so it outlives the work directory.
    publishDir { "${params.outdir}/${pair_id}/${platform.toLowerCase()}/segmentation" }, mode: "symlink", overwrite: true, pattern: "stitch_out/{proseg_latest.zarr,NO_*}"
    publishDir { "${params.outdir}/${pair_id}/${platform.toLowerCase()}/segmentation" }, mode: "copy", overwrite: true, pattern: "stitch_out/atera_proseg_report.json"

    input:
    tuple val(key), val(pair_id), val(platform), val(stitch_config_json), path(latest_zarr, stageAs: "native_latest.zarr"), path(tile_plan, stageAs: "tile_plan.json"), path(tile_dirs)

    output:
    tuple val(key),
        val(pair_id),
        val(platform),
        path("stitch_out/proseg_latest.zarr"),
        path("stitch_out/NO_CELLPOSE_MASK"),
        path("stitch_out/NO_TRANSCRIPTS_CSV"),
        path("stitch_out/NO_NUCLEI_MASK"),
        emit: segmentation
    path("stitch_out/atera_proseg_report.json"), emit: report

    script:
    """
    set -euo pipefail
    export OMP_NUM_THREADS="${task.cpus}"
    export OPENBLAS_NUM_THREADS="${task.cpus}"
    export MKL_NUM_THREADS="${task.cpus}"
    export NUMBA_NUM_THREADS="${task.cpus}"
    export POLARS_MAX_THREADS="${task.cpus}"
    export DASK_NUM_WORKERS="${task.cpus}"

    mkdir -p stitch_out

    cat > stitch_config.json <<'JSON'
${stitch_config_json}
JSON

    merxen atera-proseg-stitch --config stitch_config.json

    : > stitch_out/NO_CELLPOSE_MASK
    : > stitch_out/NO_TRANSCRIPTS_CSV
    : > stitch_out/NO_NUCLEI_MASK
    """
}
