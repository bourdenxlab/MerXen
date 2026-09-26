/*
 * CLUSTERING_MAP_FIRST: map-first clustering of one pair x segmentation
 * (plan §3.1): CLUSTERING_SQUIDPY_ANNOTATE_MAP -> _ANNOTATE_RESOLVE ->
 * _COMPUTE_CPU, emitting the input shape of CLUSTERING_SQUIDPY_FINALIZE.
 *
 * Milestone M1 scaffolding: its only caller is hook H5 in main.nf, which M5
 * adds together with these processes, so nothing here runs. Wiring it earlier
 * fails at once instead of silently emitting nothing.
 */

workflow CLUSTERING_MAP_FIRST {
    take:
    // tuple(pair_id, segmentation, samples_json, clustering_squidpy_config.json,
    // clustering_prepare_out): the output of CLUSTERING_SQUIDPY_PREPARE.
    prepared_ch

    main:
    error(
        "CLUSTERING_MAP_FIRST has no processes before milestone M5 " +
        "(docs/plans/robust-celltype-annotation-plan.md §12); " +
        "run with the default legacy clustering mode"
    )

    emit:
    prepared_ch.filter { _prepared -> false }
}
