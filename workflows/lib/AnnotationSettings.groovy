/*
 * Per-row annotation settings for main.nf (hooks H3, H6; plan §2.4, §3.1).
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
     * Summarize failed annotations and refused or provisional panels (hook H6).
     *
     * @param params Pipeline params.
     * @param runInfo Optional run facts: success (Boolean) and the lists
     *     failed_annotations, refused_panels and provisional_panels.
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
        [
            failed_annotations: "failed annotations",
            refused_panels: "refused panels",
            provisional_panels: "provisional panels",
        ].each { key, label ->
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
