/*
 * Species defaults of reference-based cell-type annotation (plan §3.7, §4.8).
 *
 * Nextflow loads the classes in workflows/lib for main.nf and every module
 * script, so the mode and table-key rules live here once (a main.nf function
 * cannot be called from a module under DSL2, which is why the clustering and
 * MapMyCells modules each re-derive the species).
 *
 * merxen.annotation.config mirrors every constant and rule in this class:
 * resolve_clustering_mode, resolve_table_key_suffix and species_defaults.
 * tests/test_workflows/test_annotation_defaults.py keeps both sides equal and
 * covers resolveMode and tableKeySuffix in all mode x flip combinations.
 */
class AnnotationDefaults {

    static final String LEGACY = "legacy"
    static final String MAP_FIRST = "map_first"
    static final List<String> MODES = [LEGACY, MAP_FIRST].asImmutable()
    static final List<String> SPECIES = ["human", "mouse"].asImmutable()

    // Species whose default clustering mode is map_first. Only the flip PRs
    // change it (human at M8 / gate H, mouse at M9 / gate M), together with
    // DEFAULT_CLUSTERING_MODE and FLIPPED_SPECIES in merxen.annotation.config.
    static final List<String> FLIPPED_SPECIES = [].asImmutable()

    // Suffix of the clustered table written by a map_first run of a species
    // that has not flipped, so the legacy table is never overwritten (OD-A3).
    static final String PRE_FLIP_TABLE_KEY_SUFFIX = "mapfirst"
    // A suffix is "" or one lower-case token (merxen.table_keys).
    static final String TABLE_KEY_SUFFIX_PATTERN = '^[a-z0-9_]*$'

    // Human regions with validated references (OD-C7: frontal cortex only).
    static final List<String> VALIDATED_HUMAN_REGIONS = ["frontal_cortex"].asImmutable()
    static final List<String> MOUSE_SECTION_REGION_KEYWORDS = ["auto", "none"].asImmutable()
    // Grey-matter CCF divisions of the mouse region model (OB split from OLF).
    static final List<String> MOUSE_CCF_REGIONS = [
        "Isocortex", "OLF", "OB", "HPF", "CTXsp", "STR", "PAL",
        "TH", "HY", "MB", "P", "MY", "CB",
    ].asImmutable()

    static final Map<String, List<String>> DEFAULT_REFERENCES = [
        human: ["whb_frontal_supc_clus", "seaad_mr_panel"].asImmutable(),
        mouse: ["wmb_panel", "wmb_region_share"].asImmutable(),
    ].asImmutable()
    static final Map<String, List<String>> KNOWN_REFERENCES = [
        human: [
            "whb_frontal_supc_clus",
            "whb_frontal_supc_clus_ho",
            "seaad_mr_panel",
            "whb_whole_ctx_panel",
            "ll_whb_ctx_profiles",
        ].asImmutable(),
        mouse: ["wmb_panel", "wmb_selfmap_testset", "wmb_region_share"].asImmutable(),
    ].asImmutable()

    private static final Map<String, String> SPECIES_ALIASES = [
        "human": "human",
        "homo_sapiens": "human",
        "homo sapiens": "human",
        "mouse": "mouse",
        "mus_musculus": "mouse",
        "mus musculus": "mouse",
    ].asImmutable()

    /**
     * Normalize a species name as main.nf's normalizeSpecies does.
     *
     * @param species Species name or alias (null means human).
     * @return "human" or "mouse".
     * @throws IllegalArgumentException On an unknown species.
     */
    static String normalizeSpecies(Object species) {
        def raw = species == null ? "human" : species.toString().trim().toLowerCase()
        if (!SPECIES_ALIASES.containsKey(raw)) {
            throw new IllegalArgumentException(
                "Unknown species '${species}'. Valid values: human, mouse"
            )
        }
        return SPECIES_ALIASES[raw]
    }

    /**
     * Return the default clustering mode of a species.
     *
     * @param species Species name or alias.
     * @param flippedSpecies Species whose default is map_first.
     * @return "map_first" for a flipped species, else "legacy".
     */
    static String defaultMode(Object species, Collection flippedSpecies = FLIPPED_SPECIES) {
        return normalizeSpecies(species) in flippedSpecies ? MAP_FIRST : LEGACY
    }

    /**
     * Resolve the clustering mode of a run (clustering_squidpy_mode*).
     *
     * clustering_squidpy_mode overrides both species; otherwise the species
     * param (clustering_squidpy_mode_human / _mouse) applies, and a null or
     * blank value falls back to the species default.
     *
     * @param params Pipeline params.
     * @param species Species name or alias.
     * @param flippedSpecies Species whose default is map_first.
     * @return "legacy" or "map_first".
     * @throws IllegalArgumentException On an unknown mode or species.
     */
    static String resolveMode(
        Map params,
        Object species,
        Collection flippedSpecies = FLIPPED_SPECIES
    ) {
        def speciesName = normalizeSpecies(species)
        def override = blankToNull(params?.get("clustering_squidpy_mode"))
        if (override != null) {
            return checkedMode(override, "clustering_squidpy_mode")
        }
        def paramName = "clustering_squidpy_mode_${speciesName}".toString()
        def perSpecies = blankToNull(params?.get(paramName))
        if (perSpecies != null) {
            return checkedMode(perSpecies, paramName)
        }
        return defaultMode(speciesName, flippedSpecies)
    }

    /**
     * Resolve the clustered table-key suffix (clustering_squidpy_table_key_suffix).
     *
     * Legacy runs always write the unsuffixed key. A map_first run uses the
     * explicit suffix when set; a null suffix means "mapfirst" while the
     * species has not flipped and "" after its flip. Before the flip an
     * explicit empty suffix is refused, because the run would overwrite the
     * legacy clustered table (OD-A3).
     *
     * @param params Pipeline params.
     * @param species Species name or alias.
     * @param flippedSpecies Species whose default is map_first.
     * @return The suffix token ("" for none).
     * @throws IllegalArgumentException On an unknown mode or species, a
     *     suffix that is not a lower-case token, or an empty suffix for a
     *     map_first run of a species that has not flipped.
     */
    static String tableKeySuffix(
        Map params,
        Object species,
        Collection flippedSpecies = FLIPPED_SPECIES
    ) {
        def speciesName = normalizeSpecies(species)
        if (resolveMode(params, speciesName, flippedSpecies) == LEGACY) {
            return ""
        }
        def explicit = params?.get("clustering_squidpy_table_key_suffix")
        if (explicit != null) {
            def token = explicit.toString().trim()
            if (!(token ==~ TABLE_KEY_SUFFIX_PATTERN)) {
                throw new IllegalArgumentException(
                    "clustering_squidpy_table_key_suffix '${explicit}' must be a " +
                    "lower-case token (letters, digits, underscores)"
                )
            }
            if (!token && !(speciesName in flippedSpecies)) {
                throw new IllegalArgumentException(emptySuffixMessage(speciesName))
            }
            return token
        }
        return speciesName in flippedSpecies ? "" : PRE_FLIP_TABLE_KEY_SUFFIX
    }

    /**
     * Return the species-dependent annotation defaults.
     *
     * Equal to merxen.annotation.config.species_defaults (string-tested).
     *
     * @param species Species name or alias.
     * @return Default clustering mode, max leaf level, spill-over flag switch,
     *     marker-consistency warning, resolvability test cells, anatomical
     *     region and reference ids.
     */
    static Map forSpecies(Object species) {
        def speciesName = normalizeSpecies(species)
        if (speciesName == "human") {
            return [
                clustering_mode: defaultMode(speciesName),
                max_leaf_level: "supercluster",
                microglial_spillover_enabled: false,
                marker_consistency_warn: 0.75,
                n_test_cells: 25000,
                anatomical_region: VALIDATED_HUMAN_REGIONS[0],
                references: DEFAULT_REFERENCES.human,
            ]
        }
        return [
            clustering_mode: defaultMode(speciesName),
            max_leaf_level: "subclass",
            microglial_spillover_enabled: true,
            marker_consistency_warn: 0.80,
            n_test_cells: 11913,
            anatomical_region: null,
            references: DEFAULT_REFERENCES.mouse,
        ]
    }

    /**
     * Explain why a map_first run of a species that has not flipped needs a suffix.
     *
     * @param species Normalized species name.
     * @return The error message (cites OD-A3).
     */
    static String emptySuffixMessage(String species) {
        return (
            "an empty clustering_squidpy_table_key_suffix would make this map_first " +
            "run write the unsuffixed clustered table and overwrite the legacy one: " +
            "${species} has not flipped to map_first yet (OD-A3, plan §4.8). Leave " +
            "the suffix unset ('${PRE_FLIP_TABLE_KEY_SUFFIX}') or name another token"
        ).toString()
    }

    private static String checkedMode(Object rawValue, String paramName) {
        def mode = rawValue.toString().trim().toLowerCase()
        if (!(mode in MODES)) {
            throw new IllegalArgumentException(
                "Unknown ${paramName} '${rawValue}'. Valid values: ${MODES.join(', ')}"
            )
        }
        return mode
    }

    private static Object blankToNull(Object value) {
        if (value == null) {
            return null
        }
        return value.toString().trim() ? value : null
    }
}
