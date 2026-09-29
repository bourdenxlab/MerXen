/*
 * Map-first clustering of one pair x segmentation (plan §3.1, §3.3).
 *
 * CLUSTERING_ANNOTATE_MAP (M3) takes the CLUSTERING_SQUIDPY_PREPARE outputs
 * through ANNOTATION_PREPARED_REFERENCES (subworkflows/annotation_references.nf,
 * M2): ANNOTATE_PANEL, one ANNOTATE_REFERENCE_PREP per unique bundle and the
 * per pair x segmentation collection, whose groupKey has that pair x
 * segmentation's own n_required. CLUSTERING_SQUIDPY_ANNOTATE_MAP therefore
 * starts as soon as the last bundle its required_bundles.json lists is
 * ready: pairs with different panels, set c or per_platform panels never
 * wait for each other, a refused panel is mapped at once with no bundle
 * (MAP records the refusal), and a pair x segmentation whose PREP failed is
 * dropped, as any failed task drops its branch under errorStrategy "ignore".
 *
 * CLUSTERING_ANNOTATE (M4) runs CLUSTERING_SQUIDPY_ANNOTATE_RESOLVE on each
 * pair x segmentation once its own MAP has finished (plan §3.4). RESOLVE is
 * a separate task from MAP: a threshold, floor, trust, flag or degraded-mode
 * change re-runs RESOLVE alone (minutes) under -resume, and MAP, whose
 * inputs and published-output reuse key hold none of them, stays cached.
 *
 * CLUSTERING_MAP_FIRST will be CLUSTERING_ANNOTATE -> COMPUTE_CPU (M5),
 * emitting the input shape of CLUSTERING_SQUIDPY_FINALIZE. Its only caller is
 * hook H5 in main.nf, which M5 adds together with COMPUTE_CPU; the preflight
 * refuses map_first runs until then (AnnotationPreflight.MAP_FIRST_WIRED), so
 * nothing here runs in a legacy or map_first pipeline run. Wiring
 * CLUSTERING_MAP_FIRST earlier fails at once instead of silently emitting
 * nothing.
 */

include { ANNOTATION_PREPARED_REFERENCES } from "./annotation_references"
include { CLUSTERING_SQUIDPY_ANNOTATE_MAP; CLUSTERING_SQUIDPY_ANNOTATE_RESOLVE } from "../modules/annotation"

workflow CLUSTERING_ANNOTATE_MAP {
    take:
    // tuple(pair_id, segmentation, samples_json, clustering_squidpy_config.json,
    // clustering_prepare_out): the output of CLUSTERING_SQUIDPY_PREPARE.
    prepared_ch

    main:
    references = ANNOTATION_PREPARED_REFERENCES(prepared_ch)

    // The map_inputs tuple is released only after the M2 required-bundle
    // join, i.e. once every bundle of the pair x segmentation is ready.
    map_inputs_ch = references.map_inputs.map { pairId, segmentation, samplesJson, clusteringConfig, preparedDir, panelDir, bundleRefs ->
        tuple(
            pairId,
            segmentation,
            AnnotationReferences.mapSpec(panelDir),
            samplesJson,
            clusteringConfig,
            preparedDir,
            panelDir,
            bundleRefs,
        )
    }
    map_out_ch = CLUSTERING_SQUIDPY_ANNOTATE_MAP(map_inputs_ch)

    // RESOLVE (M4) reads the prepared counts, the panel, the bundles and the
    // MAP outputs: join them back by pair x segmentation.
    mapped_ch = map_inputs_ch
        .map { pairId, segmentation, _mapSpec, samplesJson, clusteringConfig, preparedDir, panelDir, bundleRefs ->
            tuple(
                AnnotationReferences.branchKey(pairId, segmentation),
                pairId,
                segmentation,
                samplesJson,
                clusteringConfig,
                preparedDir,
                panelDir,
                bundleRefs,
            )
        }
        .join(
            map_out_ch.map { pairId, segmentation, mapDir ->
                tuple(AnnotationReferences.branchKey(pairId, segmentation), mapDir)
            }
        )
        .map { _branchKey, pairId, segmentation, samplesJson, clusteringConfig, preparedDir, panelDir, bundleRefs, mapDir ->
            tuple(pairId, segmentation, samplesJson, clusteringConfig, preparedDir, panelDir, bundleRefs, mapDir)
        }

    emit:
    // tuple(bundle key, bundle_ref.json): one per PREP task.
    bundle_refs = references.bundle_refs
    // tuple(pair_id, segmentation, samples_json, clustering_squidpy_config.json,
    // clustering_prepare_out, annotation_panel_out, [bundle_ref.json, ...],
    // annotation_map_out): RESOLVE's inputs (M4).
    maps = mapped_ch
}

workflow CLUSTERING_ANNOTATE {
    take:
    // tuple(pair_id, segmentation, samples_json, clustering_squidpy_config.json,
    // clustering_prepare_out): the output of CLUSTERING_SQUIDPY_PREPARE.
    prepared_ch

    main:
    mapped = CLUSTERING_ANNOTATE_MAP(prepared_ch)

    // A maps tuple exists only once its branch's MAP has finished, so each
    // RESOLVE waits for its own MAP and no other. resolveSpec carries the
    // RESOLVE config and the fingerprint of this checkout's RESOLVE rules
    // (task inputs, so -resume sees them). The shared tissue mask will come
    // from the pair's ALIGN output channel (M5, as for ANNOTATE_PANEL); until
    // then RESOLVE gets none and never looks one up among published files.
    resolve_inputs_ch = mapped.maps.map { pairId, segmentation, samplesJson, clusteringConfig, preparedDir, panelDir, bundleRefs, mapDir ->
        tuple(
            pairId,
            segmentation,
            AnnotationReferences.resolveSpec(params, panelDir, "${projectDir}/../src"),
            samplesJson,
            clusteringConfig,
            preparedDir,
            panelDir,
            bundleRefs,
            mapDir,
            [],
        )
    }
    CLUSTERING_SQUIDPY_ANNOTATE_RESOLVE(resolve_inputs_ch)
    // The deterministic annotation_resolve_out only; the run record stays
    // out of every downstream input.
    resolve_out_ch = CLUSTERING_SQUIDPY_ANNOTATE_RESOLVE.out.resolved

    labels_ch = mapped.maps
        .map { pairId, segmentation, samplesJson, clusteringConfig, preparedDir, panelDir, bundleRefs, mapDir ->
            tuple(
                AnnotationReferences.branchKey(pairId, segmentation),
                pairId,
                segmentation,
                samplesJson,
                clusteringConfig,
                preparedDir,
                panelDir,
                bundleRefs,
                mapDir,
            )
        }
        .join(
            resolve_out_ch.map { pairId, segmentation, resolveDir ->
                tuple(AnnotationReferences.branchKey(pairId, segmentation), resolveDir)
            }
        )
        .map { _branchKey, pairId, segmentation, samplesJson, clusteringConfig, preparedDir, panelDir, bundleRefs, mapDir, resolveDir ->
            tuple(pairId, segmentation, samplesJson, clusteringConfig, preparedDir, panelDir, bundleRefs, mapDir, resolveDir)
        }

    emit:
    // tuple(bundle key, bundle_ref.json): one per PREP task.
    bundle_refs = mapped.bundle_refs
    // CLUSTERING_ANNOTATE_MAP's maps (RESOLVE's inputs).
    maps = mapped.maps
    // tuple(pair_id, segmentation, samples_json, clustering_squidpy_config.json,
    // clustering_prepare_out, annotation_panel_out, [bundle_ref.json, ...],
    // annotation_map_out, annotation_resolve_out): the label tables with what
    // COMPUTE_CPU (M5) and ANNOTATION_REPORT (M7) read.
    labels = labels_ch
}

workflow CLUSTERING_MAP_FIRST {
    take:
    // tuple(pair_id, segmentation, samples_json, clustering_squidpy_config.json,
    // clustering_prepare_out): the output of CLUSTERING_SQUIDPY_PREPARE.
    prepared_ch

    main:
    // M5 replaces this guard with CLUSTERING_ANNOTATE(prepared_ch) ->
    // COMPUTE_CPU; without COMPUTE_CPU there is no FINALIZE input to emit.
    error(
        "CLUSTERING_MAP_FIRST has no COMPUTE_CPU (M5) yet " +
        "(docs/plans/robust-celltype-annotation-plan.md §12); " +
        "run with the default legacy clustering mode"
    )

    emit:
    prepared_ch.filter { _prepared -> false }
}
