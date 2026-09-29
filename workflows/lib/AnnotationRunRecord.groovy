import groovy.json.JsonSlurperClassic

import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ConcurrentSkipListSet

/*
 * What a map_first run's branches reached, for the end-of-run summary (hook
 * H6; plan §3.1 "Failure semantics", §8.5).
 *
 * Data-quality outcomes (a refused panel, a broad-only gate) are statuses
 * with outputs always written; only infrastructure errors fail a task, and
 * under errorStrategy "ignore" a failed task silently drops its pair x
 * segmentation. main.nf (hook H5) and CLUSTERING_MAP_FIRST record here, from
 * channel subscriptions in map_first runs only, which pair x segmentation
 * entered clustering, which got label tables and which got a hierarchy;
 * AnnotationSettings.completionSummary lists the branches that stopped, the
 * refused and provisional panels, the broad-only and failed dataset gates,
 * the pairs whose cross-platform statistics are restricted (by the panels or
 * by a dataset gate, as merxen.clustering.cross_platform scopes them) and
 * the samples whose MENDER run was skipped for want of an assigned cell
 * state. A legacy run subscribes nothing, so its record stays empty and it
 * prints no summary.
 *
 * One Nextflow run is one JVM, so the record is static; the subscriptions
 * run on dataflow threads, hence the concurrent collections.
 */
class AnnotationRunRecord {

    // Trust states the summary lists (plan §8.2): refused panels are also
    // listed from required_bundles.json when PANEL refused the panel itself.
    static final List<String> LISTED_TRUST_STATES = ["refused", "broad_only", "provisional"].asImmutable()

    private static final Set<String> EXPECTED = new ConcurrentSkipListSet<String>()
    private static final Set<String> LABELLED = new ConcurrentSkipListSet<String>()
    private static final Set<String> COMPUTED = new ConcurrentSkipListSet<String>()
    private static final Set<String> REFUSED_PANELS = new ConcurrentSkipListSet<String>()
    // Branches whose panel ANNOTATE_PANEL refused: their samples' refused
    // trust states are not listed again.
    private static final Set<String> REFUSED_BRANCHES = new ConcurrentSkipListSet<String>()
    private static final Map<String, Set<String>> TRUST = new ConcurrentHashMap<String, Set<String>>()
    private static final Set<String> CROSS_PLATFORM = new ConcurrentSkipListSet<String>()
    private static final Set<String> DATASET_GATES = new ConcurrentSkipListSet<String>()
    private static final Set<String> MENDER_SKIPPED = new ConcurrentSkipListSet<String>()

    // As merxen.clustering.cross_platform: statistics levels from least to
    // most restrictive, the level each dataset gate allows at most (plan
    // §5.4) and the reason a gate adds.
    static final List<String> STATISTICS_LEVELS = ["full", "broad_only", "none"].asImmutable()
    static final Map<String, String> GATE_STATISTICS_CAP = [
        full: "full",
        broad_only: "broad_only",
        failed: "none",
    ].asImmutable()
    static final String DATASET_GATE_REASON = "dataset_gate"
    // merxen.analysis.mender.SKIPPED_NO_ASSIGNED_STATE.
    static final String MENDER_SKIPPED_STATUS = "skipped_no_assigned_state"

    /** Forget everything (tests; one run per JVM otherwise). */
    static void reset() {
        [
            EXPECTED, LABELLED, COMPUTED, REFUSED_PANELS, REFUSED_BRANCHES, CROSS_PLATFORM,
            DATASET_GATES, MENDER_SKIPPED,
        ].each { Set values ->
            values.clear()
        }
        TRUST.clear()
    }

    /** Record that a pair x segmentation entered map_first clustering. */
    static void expect(Object pairId, Object segmentation) {
        EXPECTED << branch(pairId, segmentation)
    }

    /**
     * Record that a pair x segmentation got its label tables (RESOLVE done).
     *
     * @param pairId Pair id.
     * @param segmentation Segmentation.
     * @param panelDir ANNOTATE_PANEL output (required_bundles.json: status).
     * @param resolveDir RESOLVE output (the pair summary: trust and dataset
     *     gate per sample, pair.cross_platform).
     */
    static void labelled(Object pairId, Object segmentation, Object panelDir, Object resolveDir) {
        def key = branch(pairId, segmentation)
        LABELLED << key
        def required = readJson(asPath(panelDir).resolve(AnnotationReferences.REQUIRED_BUNDLES_FILE))
        if (required?.status == "refused") {
            REFUSED_BRANCHES << key
            REFUSED_PANELS << "${key}${reasonText(required.reasons)}".toString()
        }
        def summary = readJson(asPath(resolveDir).resolve(AnnotationReferences.resolveSummaryFile(pairId)))
        if (summary == null) {
            return
        }
        ((summary.samples ?: [:]) as Map).each { sampleId, sample ->
            def state = (sample instanceof Map) ? (sample.trust instanceof Map ? sample.trust.state : null) : null
            if (state == "refused" && key in REFUSED_BRANCHES) {
                return
            }
            if (state in LISTED_TRUST_STATES) {
                TRUST.computeIfAbsent(state.toString()) { ignored -> new ConcurrentSkipListSet<String>() } <<
                    "${key} ${sampleId}".toString()
            }
        }
        gateLevels(summary).each { sampleId, level ->
            if (level != "full") {
                DATASET_GATES << "${key} ${sampleId}: ${level}".toString()
            }
        }
        def scope = crossPlatformScope(summary)
        if (scope != null && scope.statistics_level != "full") {
            CROSS_PLATFORM << (
                "${key}: ${scope.statistics_level}${reasonText(scope.reasons)}"
            ).toString()
        }
    }

    /**
     * Return each sample's dataset gate level from a RESOLVE pair summary.
     *
     * @param summary The parsed pair summary.
     * @return samples[id].resolution.gate.level per sample that records one,
     *     sorted by sample id.
     */
    static Map<String, String> gateLevels(Map summary) {
        def levels = new TreeMap<String, String>()
        ((summary?.samples ?: [:]) as Map).each { sampleId, sample ->
            def resolution = (sample instanceof Map) ? sample.resolution : null
            def gate = (resolution instanceof Map) ? resolution.gate : null
            def level = (gate instanceof Map) ? gate.level : null
            if (level != null) {
                levels[sampleId.toString()] = level.toString()
            }
        }
        return levels
    }

    /**
     * Return a pair's cross-platform scope as merxen.clustering.cross_platform
     * builds it: RESOLVE's pair.cross_platform with the dataset gates folded
     * in (a broad_only gate caps the level at broad_only, a failed gate sets
     * none, each with reason dataset_gate:<sample>:<level>).
     *
     * @param summary The parsed pair summary.
     * @return [statistics_level, reasons], or null without a pair record.
     */
    static Map crossPlatformScope(Map summary) {
        def record = (summary?.pair instanceof Map) ? summary.pair.cross_platform : null
        if (!(record instanceof Map)) {
            return null
        }
        def level = (record.statistics_level ?: "none").toString()
        def reasons = ((record.reasons ?: []) as List).collect { item -> item.toString() }
        gateLevels(summary).each { sampleId, gateLevel ->
            def cap = GATE_STATISTICS_CAP[gateLevel] ?: "none"
            if (cap != "full") {
                if (STATISTICS_LEVELS.indexOf(cap) > STATISTICS_LEVELS.indexOf(level)) {
                    level = cap
                }
                reasons << "${DATASET_GATE_REASON}:${sampleId}:${gateLevel}".toString()
            }
        }
        return [statistics_level: level, reasons: reasons]
    }

    /**
     * Record one sample's MENDER import (MENDER_IMPORT done).
     *
     * @param pairId Pair id.
     * @param segmentation Segmentation.
     * @param platform Platform.
     * @param importManifest spatialdata_import_manifest.json (status).
     */
    static void menderImported(Object pairId, Object segmentation, Object platform, Object importManifest) {
        def manifest = readJson(asPath(importManifest))
        if (manifest?.status == MENDER_SKIPPED_STATUS) {
            def sample = manifest.sample_id ?: "${pairId}_${platform}"
            MENDER_SKIPPED << "${branch(pairId, segmentation)} ${sample}${reasonText(manifest.status_reasons)}".toString()
        }
    }

    // The first MAX_REASONS reasons in parentheses, with a count of the rest.
    static final int MAX_REASONS = 2

    private static String reasonText(Object reasons) {
        def items = ((reasons ?: []) as List).collect { item -> item.toString() }
        if (!items) {
            return ""
        }
        def shown = items.take(MAX_REASONS).join("; ")
        def more = items.size() > MAX_REASONS ? "; ${items.size() - MAX_REASONS} more" : ""
        return " (${shown}${more})".toString()
    }

    /** Record that a pair x segmentation got its map_first hierarchy (COMPUTE_CPU done). */
    static void computed(Object pairId, Object segmentation) {
        COMPUTED << branch(pairId, segmentation)
    }

    /**
     * Return the run facts AnnotationSettings.completionSummary lists.
     *
     * @return failed_annotations (entered clustering, no label tables),
     *     failed_hierarchies (label tables, no hierarchy), refused_panels,
     *     broad_only_panels, provisional_panels (pair:segmentation sample),
     *     restricted_dataset_gates (samples whose gate is broad_only or
     *     failed), cross_platform_restricted (pairs whose cross-platform
     *     statistics are broad_only or none, with the reasons) and
     *     mender_skipped (samples without an assigned MENDER state).
     */
    static Map runInfo() {
        return [
            n_expected: EXPECTED.size(),
            failed_annotations: (EXPECTED - LABELLED).sort(),
            failed_hierarchies: (LABELLED - COMPUTED).sort(),
            refused_panels: ((REFUSED_PANELS as List) + trustList("refused")).sort(),
            broad_only_panels: trustList("broad_only"),
            provisional_panels: trustList("provisional"),
            restricted_dataset_gates: (DATASET_GATES as List).sort(),
            cross_platform_restricted: (CROSS_PLATFORM as List).sort(),
            mender_skipped: (MENDER_SKIPPED as List).sort(),
        ]
    }

    private static List<String> trustList(String state) {
        return ((TRUST[state] ?: []) as List<String>).sort()
    }

    private static String branch(Object pairId, Object segmentation) {
        return "${pairId}:${segmentation}".toString()
    }

    private static Path asPath(Object value) {
        return value instanceof Path ? (Path) value : Paths.get(value.toString())
    }

    // Plain maps (JsonSlurperClassic): the lazy maps of JsonSlurper are not
    // safe across the dataflow threads. A missing or unreadable file gives
    // null: the summary must never fail a run.
    private static Map readJson(Path path) {
        try {
            if (!Files.isRegularFile(path)) {
                return null
            }
            def parsed = new JsonSlurperClassic().parseText(Files.readString(path))
            return parsed instanceof Map ? (Map) parsed : null
        } catch (Exception ignored) {
            return null
        }
    }
}
