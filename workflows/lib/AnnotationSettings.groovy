import groovy.json.JsonOutput
import groovy.json.JsonSlurperClassic

/*
 * Per-row annotation settings for main.nf (hooks H3, H5, H6; plan §2.4, §3.1).
 *
 * rowSampleSettings merges forRow into each row's settings map. A legacy row
 * gets no key at all: VALIDATE_ANALYSIS_LAYER takes the settings map as a
 * `val` input, so any new key would change its task hash and invalidate
 * legacy -resume. Readers therefore use the accessors below, which return
 * the legacy values when a key is absent.
 */
class AnnotationSettings {

    // Keys forRow adds to the settings of a map_first row.
    static final List<String> KEYS = [
        "clustering_squidpy_mode",
        "clustering_squidpy_table_key_suffix",
        "annotation_anatomical_region",
        "annotation_mouse_section_regions",
        "annotation_mode_mapmycells_stage",
        "mender_unassigned_state_policy",
    ].asImmutable()

    static final String LEGACY_MENDER_UNASSIGNED_STATE_POLICY = "state"
    static final String MAP_FIRST_MENDER_UNASSIGNED_STATE_POLICY = "exclude_from_features"

    // Optional samplesheet columns a map_first row carries into samples_json,
    // per species (merxen.annotation.samplesheet_columns; plan §3.3, §3.7):
    // ClusteringSquidpySampleConfig fields, None inheriting the global param.
    static final Map<String, String> ROW_COLUMNS = [
        human: "anatomical_region",
        mouse: "mouse_section_regions",
    ].asImmutable()

    // completionSummary lines: runInfo key -> label.
    static final Map<String, String> SUMMARY_ITEMS = [
        failed_annotations: "failed annotations (no label tables; see the failed tasks above)",
        failed_hierarchies: "failed hierarchies (label tables, no COMPUTE_CPU output)",
        refused_panels: "refused panels",
        broad_only_panels: "broad-only panels",
        provisional_panels: "provisional panels",
        cross_platform_restricted: "restricted cross-platform statistics",
    ].asImmutable()

    /**
     * Return the annotation settings of one samplesheet row.
     *
     * The optional samplesheet columns anatomical_region (human) and
     * mouse_section_regions (mouse) override the global params when set.
     *
     * @param row Samplesheet row.
     * @param params Pipeline params.
     * @param species Run species (name or alias).
     * @return An empty map for a legacy row, else the KEYS of a map_first row.
     * @throws IllegalArgumentException On an unknown mode or species, or an
     *     invalid table-key suffix.
     */
    static Map forRow(Map row, Map params, Object species) {
        def speciesName = AnnotationDefaults.normalizeSpecies(species)
        def mode = AnnotationDefaults.resolveMode(params, speciesName)
        if (mode == AnnotationDefaults.LEGACY) {
            return [:]
        }
        def human = speciesName == "human"
        def region = human
            ? regionToken(rowValue(row, "anatomical_region") ?: params?.get("annotation_human_region"))
            : null
        def sectionRegions = human
            ? null
            : (
                rowValue(row, "mouse_section_regions") ?:
                stringOrNull(params?.get("annotation_mouse_section_regions"))
            )
        return [
            clustering_squidpy_mode: mode,
            clustering_squidpy_table_key_suffix: AnnotationDefaults.tableKeySuffix(params, speciesName),
            annotation_anatomical_region: region,
            annotation_mouse_section_regions: sectionRegions,
            annotation_mode_mapmycells_stage: (
                stringOrNull(params?.get("annotation_mode_mapmycells_stage"))?.toLowerCase() ?:
                AnnotationDefaults.LEGACY
            ),
            mender_unassigned_state_policy: (
                stringOrNull(params?.get("mender_unassigned_state_policy")) ?:
                MAP_FIRST_MENDER_UNASSIGNED_STATE_POLICY
            ),
        ]
    }

    /**
     * Return the clustering mode recorded in a row's settings.
     *
     * @param settings Row settings from rowSampleSettings.
     * @return "legacy" when forRow added no mode, else the resolved mode.
     */
    static String mode(Map settings) {
        return settings?.get("clustering_squidpy_mode") ?: AnnotationDefaults.LEGACY
    }

    /**
     * Whether the run clusters in map_first mode (hook H5).
     *
     * The mode is one per run: it follows the run species (params.species)
     * and the clustering_squidpy_mode* params. An invalid species or mode
     * gives false here; rowSampleSettings then reports it.
     *
     * @param params Pipeline params.
     * @return Whether the resolved mode is map_first.
     */
    static boolean isMapFirstRun(Map params) {
        try {
            def species = AnnotationDefaults.normalizeSpecies(params?.get("species"))
            return AnnotationDefaults.resolveMode(params, species) == AnnotationDefaults.MAP_FIRST
        } catch (IllegalArgumentException ignored) {
            return false
        }
    }

    /**
     * Carry a map_first row's own annotation column into its samples JSON.
     *
     * Each sample gets the row's anatomical_region (human) or
     * mouse_section_regions (mouse) when the samplesheet row sets it; an
     * unset column is left out, so the sample inherits the global param
     * (ClusteringSquidpySampleConfig; plan §3.3). Legacy rows, and map_first
     * rows without the column, get the samples JSON back unchanged, so their
     * PREPARE task hash is the legacy one.
     *
     * @param samplesJson The pair x segmentation's samples JSON.
     * @param row Samplesheet row.
     * @param settings Row settings from rowSampleSettings.
     * @return The samples JSON.
     */
    static String samplesJsonWithRowColumns(Object samplesJson, Map row, Map settings) {
        if (!isMapFirst(settings)) {
            return samplesJson.toString()
        }
        def species = AnnotationDefaults.normalizeSpecies(settings.get("species"))
        def column = ROW_COLUMNS[species]
        def value = rowValue(row, column)
        if (value == null) {
            return samplesJson.toString()
        }
        def samples = new JsonSlurperClassic().parseText(samplesJson.toString()) as List
        samples.each { Map sample -> sample[column] = value }
        return JsonOutput.prettyPrint(JsonOutput.toJson(samples))
    }

    /**
     * Return the clustered-table fields a cortical-depth table config adds.
     *
     * A map_first row reads the clustered table of its own run, under the
     * run's table-key suffix (plan §4.8; CorticalDepthTableConfig
     * clustered_table_key_suffix); a legacy row adds nothing, so its config
     * JSON and task hash are unchanged.
     *
     * @param row Samplesheet row.
     * @param params Pipeline params.
     * @return [clustered_table_key_suffix: suffix] for a suffixed map_first
     *     row, else an empty map.
     */
    static Map clusteredTableFields(Map row, Map params) {
        def suffix = tableKeySuffix(forRow(row, params, params?.get("species")))
        return suffix ? [clustered_table_key_suffix: suffix] : [:]
    }

    /** Whether a row's settings select legacy clustering (the default). */
    static boolean isLegacy(Map settings) {
        return mode(settings) == AnnotationDefaults.LEGACY
    }

    /** Whether a row's settings select map_first clustering. */
    static boolean isMapFirst(Map settings) {
        return mode(settings) == AnnotationDefaults.MAP_FIRST
    }

    /**
     * Return the clustered table-key suffix of a row ("" in legacy).
     *
     * @param settings Row settings from rowSampleSettings.
     * @return The suffix token, "" when none.
     */
    static String tableKeySuffix(Map settings) {
        return settings?.get("clustering_squidpy_table_key_suffix") ?: ""
    }

    /**
     * Apply the map_first MAPMYCELLS rule to the stage selection (plan §3.1).
     *
     * A map_first row runs the legacy MAPMYCELLS stage only when
     * annotation_mode_mapmycells_stage is "legacy" and the user stops at
     * mapmycells (--stop_stage or --only_stage mapmycells). Legacy rows keep
     * the stage selection unchanged.
     *
     * @param stageSelected Whether mapmycells lies in the selected stage range.
     * @param annotationSettings The row's forRow result.
     * @param stopStage The row's normalized stop stage.
     * @return Whether MAPMYCELLS runs for the row.
     */
    static boolean runMapMyCells(
        boolean stageSelected,
        Map annotationSettings,
        Object stopStage
    ) {
        if (!isMapFirst(annotationSettings)) {
            return stageSelected
        }
        return (
            stageSelected &&
            annotationSettings.get("annotation_mode_mapmycells_stage") == AnnotationDefaults.LEGACY &&
            stopStage?.toString() == "mapmycells"
        )
    }

    /**
     * Summarize a map_first run's failed branches and panels (hook H6).
     *
     * @param params Pipeline params.
     * @param runInfo Optional run facts: success (Boolean), n_expected and
     *     the SUMMARY_ITEMS lists (AnnotationRunRecord.runInfo).
     * @return "" when the run's species uses legacy mode (nothing to report),
     *     else the summary text.
     */
    static String completionSummary(Map params, Map runInfo = [:]) {
        def species
        def mode
        try {
            species = AnnotationDefaults.normalizeSpecies(params?.get("species"))
            mode = AnnotationDefaults.resolveMode(params, species)
        } catch (IllegalArgumentException ignored) {
            // rowSampleSettings has already reported the invalid param.
            return ""
        }
        if (mode == AnnotationDefaults.LEGACY) {
            return ""
        }
        def lines = ["Annotation summary (${species}, clustering_squidpy_mode ${mode}):".toString()]
        if (!AnnotationPreflight.MAP_FIRST_WIRED) {
            lines << (
                "  map_first is not wired in this version (M1 scaffolding); " +
                "no annotation task ran."
            )
        }
        if (runInfo?.get("n_expected") != null) {
            lines << "  pair x segmentation branches clustered: ${runInfo.n_expected}".toString()
        }
        SUMMARY_ITEMS.each { key, label ->
            def values = (runInfo?.get(key) ?: []) as List
            lines << "  ${label}: ${values ? values.join(', ') : 'none'}".toString()
        }
        return lines.join("\n")
    }

    private static String rowValue(Map row, String column) {
        return stringOrNull(row?.get(column))
    }

    private static String stringOrNull(Object value) {
        if (value == null) {
            return null
        }
        def text = value.toString().trim()
        return text ?: null
    }

    // As merxen.annotation.samplesheet_columns.parse_anatomical_region.
    private static String regionToken(Object value) {
        def text = stringOrNull(value)
        return text == null ? null : text.toLowerCase().replaceAll(/[\s\-]+/, "_")
    }
}
