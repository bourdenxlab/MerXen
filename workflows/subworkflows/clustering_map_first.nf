/*
 * CLUSTERING_MAP_FIRST: map-first clustering of one pair x segmentation
 * (plan §3.1): CLUSTERING_SQUIDPY_ANNOTATE_MAP -> _ANNOTATE_RESOLVE ->
 * _COMPUTE_CPU, emitting the input shape of CLUSTERING_SQUIDPY_FINALIZE.
 *
 * MAP takes its reference bundles from ANNOTATION_PREPARED_REFERENCES
 * (subworkflows/annotation_references.nf, M2): ANNOTATE_PANEL and
 * ANNOTATE_REFERENCE_PREP on these prepared outputs, emitting MAP's input
 * tuple once every bundle of the pair x segmentation is ready.
 *
 * Its only caller is hook H5 in main.nf, which M5 adds together with MAP
 * (M3), RESOLVE (M4) and COMPUTE_CPU (M5), so nothing here runs. Wiring it
 * earlier fails at once instead of silently emitting nothing.
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
