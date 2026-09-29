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
 * refused and provisional panels and the pairs whose cross-platform
 * statistics RESOLVE restricted. A legacy run subscribes nothing, so its
 * record stays empty and it prints no summary.
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

    /** Forget everything (tests; one run per JVM otherwise). */
    static void reset() {
        [EXPECTED, LABELLED, COMPUTED, REFUSED_PANELS, REFUSED_BRANCHES, CROSS_PLATFORM].each { Set values ->
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
     * @param resolveDir RESOLVE output (the pair summary: trust per sample,
     *     pair.cross_platform).
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
        def crossPlatform = (summary.pair instanceof Map) ? summary.pair.cross_platform : null
        if (crossPlatform instanceof Map && crossPlatform.statistics_level != "full") {
            CROSS_PLATFORM << (
                "${key}: ${crossPlatform.statistics_level}${reasonText(crossPlatform.reasons)}"
            ).toString()
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
     *     broad_only_panels, provisional_panels (pair:segmentation sample)
     *     and cross_platform_restricted (pairs whose cross-platform
     *     statistics are broad_only or none, with RESOLVE's reasons).
     */
    static Map runInfo() {
        return [
            n_expected: EXPECTED.size(),
            failed_annotations: (EXPECTED - LABELLED).sort(),
            failed_hierarchies: (LABELLED - COMPUTED).sort(),
            refused_panels: ((REFUSED_PANELS as List) + trustList("refused")).sort(),
            broad_only_panels: trustList("broad_only"),
            provisional_panels: trustList("provisional"),
            cross_platform_restricted: (CROSS_PLATFORM as List).sort(),
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
