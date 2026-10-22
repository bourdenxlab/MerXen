/*
 * Start-up check of the private site configs (docs/local-configuration.md).
 *
 * The host's locations (reference stores, sources and caches, the Cellpose
 * model store, the task temporary directory, the container image and bind
 * mounts) live in two gitignored site configs that nextflow.config and
 * conf/dwight.config include. When one is missing, several params stay null
 * and the run does NOT fail on them: it falls back silently. A null
 * annotation_reference_store becomes <outdir>/annotation_references, so PREP
 * rebuilds every reference bundle (about 40 min and 48-57 GB each) instead of
 * reusing the store; the reference caches fall back to <outdir> and the
 * reference downloads default to on, so null cache directories mean
 * downloads; with no TMPDIR the tasks fill the small root disk; and a null
 * apptainer image runs the tasks without a container. This check turns each
 * of those into an error at launch, unless the operator switched the site
 * config off on purpose (MERXEN_DWIGHT_SITE_CONFIG / MERXEN_SITE_CONFIG set to
 * 'none' or /dev/null; an empty value counts as unset, as everywhere else).
 *
 * Pure function of its arguments (main.nf passes workflow.profile,
 * workflow.containerEngine, params and the environment), so the Groovy tests
 * run it on many cases in one Nextflow run.
 */
class SiteConfigCheck {

    // Profiles that include conf/dwight.config, and so conf/dwight.site.config:
    // `standard` is the profile Nextflow selects when -profile is not given.
    static final List<String> DWIGHT_PROFILES = ["standard", "dwight", "local"].asImmutable()

    // Values of the site-config variables that mean "absent on purpose".
    static final List<String> ABSENT = ["none", "/dev/null"].asImmutable()

    /**
     * Return the start-up errors of a launch (empty when it may start).
     *
     * @param profile the launch's profile string (workflow.profile, e.g. "dwight,conda")
     * @param containerEngine the enabled container engine (workflow.containerEngine) or null
     * @param params the run's params
     * @param env the process environment (System.getenv())
     */
    static List<String> problems(Object profile, Object containerEngine, Map params, Map env) {
        List<String> errors = []
        List<String> profiles = (profile == null ? "" : profile.toString())
            .split(",")
            .collect { name -> name.trim() }
            .findAll { name -> name }
        if (profiles.any { name -> DWIGHT_PROFILES.contains(name) } &&
            isBlank(params.task_tmp_dir) &&
            !isAbsentOnPurpose(env, "MERXEN_DWIGHT_SITE_CONFIG")) {
            errors << (
                "no Dwight site config (params.task_tmp_dir is not set): set MERXEN_DWIGHT_SITE_CONFIG, " +
                "install workflows/conf/dwight.site.config, or MERXEN_DWIGHT_SITE_CONFIG=none to run " +
                "without host locations (docs/local-configuration.md)"
            )
        }
        if (containerEngine != null && containerEngine.toString() == "apptainer" &&
            isBlank(params.apptainer_container) &&
            !isAbsentOnPurpose(env, "MERXEN_SITE_CONFIG")) {
            errors << (
                "no site config for the apptainer profile (params.apptainer_container is not set: the " +
                "tasks would run without a container): set MERXEN_SITE_CONFIG, install " +
                "workflows/conf/site.config, or MERXEN_SITE_CONFIG=none to run without a container " +
                "image (docs/local-configuration.md)"
            )
        }
        return errors
    }

    private static boolean isBlank(Object value) {
        return value == null || value.toString().trim().isEmpty()
    }

    private static boolean isAbsentOnPurpose(Map env, String name) {
        def value = env[name]
        return value != null && ABSENT.contains(value.toString())
    }
}
