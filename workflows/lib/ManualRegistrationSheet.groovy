import groovy.json.JsonOutput
import groovy.json.JsonSlurper

import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths

/*
 * Samplesheet reader and config emitter of the standalone manual-registration
 * workflows (workflows/manual_registration_{prepare,register}.nf;
 * docs/stages/manual-registration.md).
 *
 * One samplesheet (one row per image pair) feeds both workflows. This class
 * validates every column, collects ALL errors of a sheet into one list (the
 * workflows print them as one "Preflight checks failed:" block before any
 * task is created), checks the prepared stores the register workflow needs,
 * and builds the JSON config of every task. The Python commands validate the
 * same configs again with pydantic (tests/test_workflows keep the key sets in
 * step with the models).
 *
 * Pure: no Nextflow API (no file(), log or params globals); everything is
 * passed in, so the Groovy tests run it on many cases in one Nextflow launch
 * (tests/test_workflows/test_manual_registration_sheet.py).
 */
class ManualRegistrationSheet {

    static final String ID_PATTERN = '^[A-Za-z0-9][A-Za-z0-9_.-]*$'
    static final List<String> PRESETS = ["same_section", "serial_section"].asImmutable()
    static final List<String> REQUIRED_COLUMNS = ["id", "reference_image", "moving_image"].asImmutable()
    static final List<String> ALLOWED_COLUMNS = [
        "id", "reference_image", "moving_image",
        "reference_scene", "moving_scene",
        "reference_channel_names", "moving_channel_names",
        "reference_registration_channel", "moving_registration_channel",
        "reference_pixel_size_um", "moving_pixel_size_um",
        "presets", "registration_settings", "run_label", "prepared_store",
    ].asImmutable()
    static final List<String> IMAGE_EXTENSIONS = [".czi", ".ome.tif", ".ome.tiff", ".tif", ".tiff"].asImmutable()
    static final List<String> FORBIDDEN_SETTINGS = ["reference_pixel_size_um", "reference_channel"].asImmutable()
    // = sorted(set(ManualRegistrationSettings.model_fields) - FORBIDDEN_OVERRIDES); a test keeps them equal.
    static final List<String> SETTINGS_KEYS = [
        "build_viewer_pyramid",
        "chunk_size",
        "crop_min_size_px",
        "crop_snap_px",
        "landmark_max_outlier_fraction",
        "landmark_min_pairs",
        "landmark_min_spread_ratio",
        "landmark_min_used_pairs",
        "landmark_outlier_floor_um",
        "landmark_outlier_k",
        "landmark_outlier_policy",
        "landmark_reflection_check_ratio",
        "landmarks_outside_crop",
        "matrix_scale_tolerance",
        "max_refinement_rotation_deg",
        "max_refinement_scale_change",
        "max_refinement_shift_um",
        "min_refinement_windows",
        "nonrigid_attachment",
        "nonrigid_enabled",
        "nonrigid_highpass_sigma_um",
        "nonrigid_local_correlation_sigma_um",
        "nonrigid_local_residual_tile_um",
        "nonrigid_local_residual_tiles",
        "nonrigid_max_cycle_error_um",
        "nonrigid_max_displacement_um",
        "nonrigid_max_jacobian",
        "nonrigid_min_confident_fraction",
        "nonrigid_min_jacobian",
        "nonrigid_min_local_correlation",
        "nonrigid_min_residual_correlation_improvement",
        "nonrigid_min_residual_initial_confidence",
        "nonrigid_num_iter",
        "nonrigid_num_warp",
        "nonrigid_pixel_size_um",
        "nonrigid_residual_passes",
        "nonrigid_smoothing_sigma_um",
        "nonrigid_tightness",
        "nucleus_qc",
        "nucleus_qc_pixel_size_um",
        "nucleus_qc_window_um",
        "refine_affine",
        "refinement_min_window_quality",
        "refinement_require_moving_tissue",
        "refinement_window_um",
        "registration_pixel_size_um",
        "tile_size",
        "tissue_qc_highpass_sigma_um",
        "tissue_qc_pixel_size_um",
        "tissue_qc_sigma_um",
    ].asImmutable()
    static final int MIN_LANDMARK_PAIRS = 8
    static final int SCHEMA_VERSION = 1
    static final String STORE_ATTR = "merxen_manual_registration_store"
    static final String LABELLING_ATTR = "merxen_manual_registration"

    static final String PRESET_LIST = "same_section, serial_section"

    // ------------------------------------------------------------------
    // Small helpers
    // ------------------------------------------------------------------

    /** Value of an on/off param: true/"true"/"yes"/"1"/1; false for false/"false"/"no"/"0"/0/null/"". */
    static boolean flag(Object value) {
        if (value == null) {
            return false
        }
        if (value instanceof Boolean) {
            return (boolean) value
        }
        String text = value.toString().trim().toLowerCase()
        if (text in ["true", "yes", "1"]) {
            return true
        }
        if (text in ["false", "no", "0", ""]) {
            return false
        }
        throw new IllegalArgumentException("not a boolean: '${value}'".toString())
    }

    /** Trimmed --manual_registration_czi_python, or null when blank. */
    static String cziPython(Map params) {
        Object value = params?.manual_registration_czi_python
        if (value == null) {
            return null
        }
        String text = value.toString().trim()
        return text ? text : null
    }

    /** Absolute, normalised --outdir without a trailing slash. */
    static String outdirAbs(Map params) {
        return Paths.get(params.outdir.toString()).toAbsolutePath().normalize().toString()
    }

    static String preparedDir(Map params, String id) {
        return "${outdirAbs(params)}/${id}/prepared".toString()
    }

    static String preparedStorePath(Map params, Map row) {
        if (row.prepared_store) {
            return row.prepared_store.toString()
        }
        return "${preparedDir(params, row.id.toString())}/${row.id}_prepared.zarr".toString()
    }

    static String runName(String preset, String runLabel) {
        return (runLabel != null && runLabel.trim()) ? "${preset}_${runLabel.trim()}".toString() : preset
    }

    static String runId(String id, String preset, String runLabel) {
        return "${id}_${runName(preset, runLabel)}".toString()
    }

    static String runDir(Map params, String id, String preset, String runLabel) {
        return "${outdirAbs(params)}/${id}/registered/${runName(preset, runLabel)}".toString()
    }

    static String summaryPath(Map params, String id, String preset, String runLabel) {
        return "${runDir(params, id, preset, runLabel)}/${runId(id, preset, runLabel)}_manual_registration_summary.json".toString()
    }

    static String resultStorePath(Map params, String id, String preset, String runLabel) {
        return "${runDir(params, id, preset, runLabel)}/${runId(id, preset, runLabel)}_registered.zarr".toString()
    }

    static String comparisonCsvPath(Map params, String id) {
        return "${outdirAbs(params)}/${id}/registered/${id}_run_comparison.csv".toString()
    }

    // ------------------------------------------------------------------
    // CSV
    // ------------------------------------------------------------------

    /**
     * RFC 4180 reader: "," separator, quoted fields (a doubled quote is a literal quote; commas,
     * semicolons and newlines are literal inside quotes), records end at LF, CRLF or a lone CR.
     * Cells are trimmed; records whose cells are all blank are dropped.
     *
     * @throws IllegalArgumentException on an unterminated quoted field
     */
    static List<List<String>> parseCsv(String text) {
        List<List<String>> records = []
        List<String> row = []
        StringBuilder cell = new StringBuilder()
        boolean inQuotes = false
        boolean fieldStart = true
        int quoteRecord = 0
        int i = 0
        int n = text.length()
        while (i < n) {
            char c = text.charAt(i)
            if (inQuotes) {
                if (c == (char) '"') {
                    if (i + 1 < n && text.charAt(i + 1) == (char) '"') {
                        cell.append('"')
                        i += 1
                    } else {
                        inQuotes = false
                    }
                } else {
                    cell.append(c)
                }
            } else if (c == (char) '"' && fieldStart) {
                inQuotes = true
                fieldStart = false
                quoteRecord = records.size() + 1
            } else if (c == (char) ',') {
                row.add(cell.toString().trim())
                cell = new StringBuilder()
                fieldStart = true
            } else if (c == (char) '\n' || c == (char) '\r') {
                if (c == (char) '\r' && i + 1 < n && text.charAt(i + 1) == (char) '\n') {
                    i += 1
                }
                row.add(cell.toString().trim())
                cell = new StringBuilder()
                fieldStart = true
                if (row.any { String v -> v }) {
                    records.add(row)
                } else {
                    records.add([])
                }
                row = []
            } else {
                if (!Character.isWhitespace(c)) {
                    fieldStart = false
                }
                cell.append(c)
            }
            i += 1
        }
        if (inQuotes) {
            throw new IllegalArgumentException(
                "samplesheet: unterminated quoted field starting in row ${quoteRecord}".toString()
            )
        }
        if (cell.length() > 0 || !row.isEmpty()) {
            row.add(cell.toString().trim())
            records.add(row.any { String v -> v } ? row : [])
        }
        return records.findAll { List<String> r -> !r.isEmpty() }
    }

    // ------------------------------------------------------------------
    // Settings grammar
    // ------------------------------------------------------------------

    /** Items of "[preset:]key=value;..." text: [raw, preset, key, value] or [raw, null, null, null] when malformed. */
    static List<Map> settingsItems(String text) {
        List<Map> items = []
        if (text == null || !text.trim()) {
            return items
        }
        List<String> parts = text.split(";", -1).collect { String p -> p.trim() }
        if (parts && parts[-1] == "") {
            parts = parts.subList(0, parts.size() - 1)
        }
        for (String item : parts) {
            Map entry = [raw: item, preset: null, key: null, value: null, empty: item == "", malformed: false]
            int eq = item.indexOf("=")
            if (item == "" || eq < 0) {
                entry.malformed = true
            } else {
                String left = item.substring(0, eq).trim()
                entry.value = item.substring(eq + 1).trim()
                int colon = left.indexOf(":")
                if (colon >= 0) {
                    entry.preset = left.substring(0, colon).trim()
                    entry.key = left.substring(colon + 1).trim()
                } else {
                    entry.key = left
                }
            }
            items.add(entry)
        }
        return items
    }

    static List<String> settingsErrors(String text, String label) {
        List<String> errors = []
        Set<String> seen = new HashSet<String>()
        for (Map item : settingsItems(text)) {
            String raw = item.raw
            if (item.empty) {
                errors.add("${label}: empty item (stray ';') in '${text.trim()}'".toString())
                continue
            }
            if (item.malformed) {
                errors.add("${label}: item '${raw}': expected [preset:]key=value".toString())
                continue
            }
            String preset = item.preset
            String key = item.key
            String value = item.value
            if (preset != null && !(preset in PRESETS)) {
                errors.add("${label}: item '${raw}': unknown preset prefix '${preset}' (allowed: ${PRESET_LIST})".toString())
                continue
            }
            if (!(key ==~ '^[a-z][a-z0-9_]*$')) {
                errors.add("${label}: item '${raw}': key '${key}' must match [a-z][a-z0-9_]*".toString())
                continue
            }
            if (key in FORBIDDEN_SETTINGS) {
                errors.add(
                    "${label}: '${key}' cannot be overridden: set pixel sizes / registration channels in the samplesheet and re-run prepare".toString()
                )
                continue
            }
            if (!(key in SETTINGS_KEYS)) {
                errors.add("${label}: unknown key '${key}' (allowed: ${SETTINGS_KEYS.join(', ')})".toString())
                continue
            }
            if (!value) {
                errors.add("${label}: item '${raw}': empty value for '${key}'".toString())
                continue
            }
            String scope = preset ?: "all presets"
            if (!seen.add("${scope}|${key}".toString())) {
                errors.add("${label}: '${key}' is given twice for scope ${scope}".toString())
                continue
            }
            if (key == "landmark_min_pairs") {
                boolean ok = false
                if (value ==~ '^[0-9]{1,9}$') {
                    ok = Integer.parseInt(value) >= 3
                }
                if (!ok) {
                    errors.add("${label}: landmark_min_pairs must be an integer >= 3 (got '${value}')".toString())
                }
            }
        }
        return errors
    }

    /**
     * landmark_min_pairs in force for one run: global unprefixed < global preset: < row unprefixed < row
     * preset: (the order of resolve_settings in Python); the last assignment wins, default 8.
     */
    static Integer effectiveMinPairs(Map row, String preset, Map params) {
        Integer result = MIN_LANDMARK_PAIRS
        List<String> sources = [
            (params?.manual_registration_settings ?: "").toString(),
            (row?.registration_settings ?: "").toString(),
        ]
        for (String text : sources) {
            List<Map> items = settingsItems(text)
            for (boolean prefixed : [false, true]) {
                for (Map item : items) {
                    if (item.malformed || item.key != "landmark_min_pairs") {
                        continue
                    }
                    boolean isPrefixed = item.preset != null
                    if (isPrefixed != prefixed || (isPrefixed && item.preset != preset)) {
                        continue
                    }
                    String value = item.value
                    if (value ==~ '^[0-9]{1,9}$') {
                        result = Integer.parseInt(value)
                    }
                }
            }
        }
        return result
    }

    // ------------------------------------------------------------------
    // parse
    // ------------------------------------------------------------------

    /** Split and validate a ";"-list of presets: [presets, errors]. */
    static List parsePresets(String text, String label) {
        List<String> errors = []
        List<String> presets = []
        List<String> items = (text ?: "").split(";", -1).collect { String p -> p.trim() }.findAll { String p -> p }
        if (!items) {
            errors.add("${label}: no preset given".toString())
            return [presets, errors]
        }
        for (String item : items) {
            if (!(item in PRESETS)) {
                errors.add("${label}: unknown preset '${item}' (allowed: ${PRESET_LIST})".toString())
            } else if (item in presets) {
                errors.add("${label}: '${item}' listed twice".toString())
            } else {
                presets.add(item)
            }
        }
        return [presets, errors]
    }

    static Path resolveInput(Path sheet, String value) {
        Path candidate = Paths.get(value)
        if (candidate.isAbsolute()) {
            return candidate.normalize()
        }
        return sheet.toAbsolutePath().parent.resolve(value).normalize()
    }

    /**
     * Read and validate one samplesheet.
     *
     * @param sheet the samplesheet file
     * @param params the run's params (outdir, manual_registration_*)
     * @param mode "prepare" or "register"
     * @return [rows: normalised rows, errors: all error messages, warnings: all warnings]
     */
    static Map parse(Path sheet, Map params, String mode) {
        List<String> errors = []
        List<String> warnings = []
        List<Map> rows = []
        Map result = [rows: rows, errors: errors, warnings: warnings]

        // ---- params -------------------------------------------------
        List<String> defaultPresets = ["same_section"]
        if (!(params?.outdir?.toString()?.trim())) {
            errors.add("--outdir is required")
        }
        List presetCheck = parsePresets((params?.manual_registration_presets ?: "").toString(), "--manual_registration_presets")
        errors.addAll((List<String>) presetCheck[1])
        if (!((List<String>) presetCheck[1])) {
            defaultPresets = (List<String>) presetCheck[0]
        }
        errors.addAll(settingsErrors((params?.manual_registration_settings ?: "").toString(), "--manual_registration_settings"))
        for (String name : ["manual_registration_force_prepare", "manual_registration_force_register"]) {
            try {
                flag(params?.get(name))
            } catch (IllegalArgumentException ignored) {
                errors.add("--${name} must be true or false (got '${params?.get(name)}')".toString())
            }
        }
        String czi = cziPython(params)
        if (mode == "prepare" && czi != null && czi != "in_process") {
            if (!Paths.get(czi).isAbsolute()) {
                errors.add("--manual_registration_czi_python must be an absolute path (got '${czi}')".toString())
            } else if (czi.contains("'")) {
                errors.add("--manual_registration_czi_python must not contain a quote")
            } else if (!Files.isRegularFile(Paths.get(czi))) {
                errors.add("--manual_registration_czi_python not found: ${czi}".toString())
            }
        }

        // ---- csv ----------------------------------------------------
        List<List<String>> records
        try {
            String text = new String(Files.readAllBytes(sheet), StandardCharsets.UTF_8)
            if (text.startsWith("﻿")) {
                text = text.substring(1)
            }
            records = parseCsv(text)
        } catch (IllegalArgumentException e) {
            errors.add(e.message)
            return result
        }
        if (records.isEmpty()) {
            errors.add("samplesheet: no rows")
            return result
        }
        List<String> header = records[0]
        Set<String> seenColumns = new LinkedHashSet<String>()
        List<String> duplicated = []
        header.eachWithIndex { String name, int index ->
            if (name == "") {
                errors.add("samplesheet: column ${index + 1} has an empty header name".toString())
            } else if (!seenColumns.add(name) && !(name in duplicated)) {
                duplicated.add(name)
            }
        }
        duplicated.each { String name -> errors.add("samplesheet: column '${name}' appears twice".toString()) }
        List<String> missing = REQUIRED_COLUMNS.findAll { String c -> !(c in seenColumns) }
        if (missing) {
            errors.add(
                "samplesheet: missing required column(s): ${missing.join(', ')} (required: ${REQUIRED_COLUMNS.join(', ')})".toString()
            )
        }
        List<String> unknown = seenColumns.findAll { String c -> !(c in ALLOWED_COLUMNS) }.toList()
        if (unknown) {
            errors.add("samplesheet: unknown column(s): ${unknown.join(', ')} (allowed: ${ALLOWED_COLUMNS.join(', ')})".toString())
        }
        if (records.size() < 2) {
            errors.add("samplesheet: no rows")
        }
        int headerErrors = errors.count { String e -> e.startsWith("samplesheet:") }
        if (headerErrors > 0) {
            return result
        }

        // ---- rows ---------------------------------------------------
        Map<String, Integer> idRows = [:]
        for (int k = 1; k < records.size(); k++) {
            List<String> cells = records[k]
            int n = k + 1
            if (cells.size() > header.size() && cells.subList(header.size(), cells.size()).any { String v -> v }) {
                errors.add("row ${n}: ${cells.size()} cells but the header has ${header.size()}".toString())
                continue
            }
            Map<String, String> r = [:]
            header.eachWithIndex { String name, int index ->
                String value = index < cells.size() ? cells[index] : ""
                r[name] = value
            }
            Map row = validateRow(r, n, sheet, params, mode, defaultPresets, idRows, errors, warnings)
            if (row != null) {
                rows.add(row)
            }
        }

        // ---- the CZI runtime is needed by prepare only -------------------
        if (mode == "prepare" && czi == null) {
            for (Map row : rows) {
                List<String> czisRoles = []
                for (String role : ["reference", "moving"]) {
                    if (row[role].path.toString().toLowerCase().endsWith(".czi")) {
                        czisRoles.add("${role}_image".toString())
                    }
                }
                if (czisRoles) {
                    errors.add(
                        ("row ${row.row_number} (${row.id}): ${czisRoles.join(' and ')} is a .czi file but " +
                        "--manual_registration_czi_python is not set; set it in workflows/conf/site.config to the " +
                        "python of the CZI runtime (envs/environment.czi.yml; see docs/stages/manual-registration.md), " +
                        "or to in_process if czifile is installed in the task environment").toString()
                    )
                }
            }
        }
        return result
    }

    /** Validate one record; returns the normalised row (null when the row has errors). */
    static Map validateRow(
        Map<String, String> r, int n, Path sheet, Map params, String mode,
        List<String> defaultPresets, Map<String, Integer> idRows,
        List<String> errors, List<String> warnings
    ) {
        List<String> rowErrors = []
        String id = r.id ?: ""
        boolean idValid = id != "" && (id ==~ ID_PATTERN)
        String prefix = "row ${n} (${idValid ? id : '?'}): "

        if (id == "") {
            rowErrors.add("id is required")
        } else if (!idValid) {
            rowErrors.add("id '${id}' must match ${ID_PATTERN}".toString())
        } else if (idRows.containsKey(id)) {
            rowErrors.add("id '${id}' is also used in row ${idRows[id]}".toString())
        } else {
            idRows[id] = n
        }

        Map<String, Path> paths = [:]
        Map<String, Boolean> exists = [:]
        for (String role : ["reference", "moving"]) {
            String column = "${role}_image".toString()
            String value = r[column] ?: ""
            if (value == "") {
                rowErrors.add("${column} is required".toString())
                continue
            }
            Path path
            try {
                path = resolveInput(sheet, value)
            } catch (java.nio.file.InvalidPathException ignored) {
                rowErrors.add("${column} is not a valid path: ${value}".toString())
                continue
            }
            paths[role] = path
            String lower = path.getFileName().toString().toLowerCase()
            if (!IMAGE_EXTENSIONS.any { String ext -> lower.endsWith(ext) }) {
                rowErrors.add("${column} must end in .czi, .ome.tif, .ome.tiff, .tif or .tiff: ${path}".toString())
            }
            if (Files.isRegularFile(path)) {
                exists[role] = true
            } else {
                rowErrors.add("${column} not found: ${path}".toString())
            }
        }
        if (exists.reference && exists.moving) {
            if (paths.reference.toRealPath() == paths.moving.toRealPath()) {
                rowErrors.add("reference_image and moving_image are the same file")
            }
        }

        Map<String, Map> inputs = [:]
        for (String role : ["reference", "moving"]) {
            Map input = [path: paths[role]?.toString(), scene: null, channel_names: null, registration_channel: "DAPI", pixel_size_um: null]
            String scene = r["${role}_scene".toString()] ?: ""
            if (scene != "") {
                if (paths[role] != null && !paths[role].getFileName().toString().toLowerCase().endsWith(".czi")) {
                    rowErrors.add("${role}_scene is only valid for a .czi ${role}_image".toString())
                } else {
                    input.scene = scene
                }
            }
            String names = r["${role}_channel_names".toString()] ?: ""
            if (names != "") {
                List<String> items = names.split(";", -1).collect { String p -> p.trim() }
                boolean ok = true
                Set<String> lowered = new HashSet<String>()
                for (String item : items) {
                    if (item == "") {
                        rowErrors.add("${role}_channel_names has an empty name".toString())
                        ok = false
                    } else if (!lowered.add(item.toLowerCase())) {
                        rowErrors.add("${role}_channel_names repeats '${item}'".toString())
                        ok = false
                    }
                }
                if (ok) {
                    input.channel_names = items
                }
            }
            String channel = r["${role}_registration_channel".toString()] ?: ""
            if (channel != "") {
                input.registration_channel = channel
            }
            String size = r["${role}_pixel_size_um".toString()] ?: ""
            if (size != "") {
                try {
                    BigDecimal parsed = new BigDecimal(size)
                    if (parsed.signum() <= 0) {
                        rowErrors.add("${role}_pixel_size_um must be a number > 0 (got '${size}')".toString())
                    } else {
                        input.pixel_size_um = parsed.doubleValue()
                    }
                } catch (NumberFormatException ignored) {
                    rowErrors.add("${role}_pixel_size_um must be a number > 0 (got '${size}')".toString())
                }
            }
            inputs[role] = input
        }

        List<String> presets = defaultPresets
        String presetText = r.presets ?: ""
        if (presetText != "") {
            List check = parsePresets(presetText, "presets")
            rowErrors.addAll((List<String>) check[1])
            presets = (List<String>) check[0]
        }

        String settings = (r.registration_settings ?: "").trim()
        rowErrors.addAll(settingsErrors(settings, "registration_settings"))

        String runLabel = r.run_label ?: ""
        if (runLabel != "" && !(runLabel ==~ ID_PATTERN)) {
            rowErrors.add("run_label '${runLabel}' must match ${ID_PATTERN}".toString())
        }

        String defaultStore = null
        if (idValid && params?.outdir?.toString()?.trim()) {
            defaultStore = "${preparedDir(params, id)}/${id}_prepared.zarr".toString()
        }
        String storeValue = r.prepared_store ?: ""
        String preparedStore = defaultStore
        if (storeValue != "") {
            try {
                String resolved = resolveInput(sheet, storeValue).toString()
                if (mode == "register") {
                    preparedStore = resolved
                } else if (resolved != defaultStore) {
                    warnings.add(
                        "row ${n} (${idValid ? id : '?'}): prepared_store is read by the register workflow only; prepare writes ${defaultStore}".toString()
                    )
                }
            } catch (java.nio.file.InvalidPathException ignored) {
                rowErrors.add("prepared_store is not a valid path: ${storeValue}".toString())
            }
        }

        if (rowErrors) {
            rowErrors.each { String e -> errors.add((prefix + e).toString()) }
            return null
        }
        return [
            row_number: n,
            id: id,
            reference: inputs.reference,
            moving: inputs.moving,
            presets: presets,
            registration_settings: settings,
            run_label: runLabel != "" ? runLabel : null,
            prepared_store: preparedStore,
        ]
    }

    // ------------------------------------------------------------------
    // Prepared stores
    // ------------------------------------------------------------------

    /**
     * Root attributes of a store, read without Zarr: zarr.json "attributes" (v3) or .zattrs (v2).
     *
     * @return [exists, kind, schemaVersion, status, labellingSchemaVersion, nEnabledPairs, nPairs, error]
     */
    static Map storeState(String storePath) {
        Map state = [
            exists: false, kind: null, schemaVersion: null, status: null,
            labellingSchemaVersion: null, nEnabledPairs: 0, nPairs: 0, error: null,
        ]
        Path store = Paths.get(storePath)
        if (!Files.isDirectory(store)) {
            return state
        }
        state.exists = true
        try {
            Map attrs
            Path v3 = store.resolve("zarr.json")
            Path v2 = store.resolve(".zattrs")
            if (Files.isRegularFile(v3)) {
                Object doc = new JsonSlurper().parse(v3.toFile())
                attrs = (doc instanceof Map && doc.attributes instanceof Map) ? (Map) doc.attributes : [:]
            } else if (Files.isRegularFile(v2)) {
                Object doc = new JsonSlurper().parse(v2.toFile())
                attrs = (doc instanceof Map) ? (Map) doc : [:]
            } else {
                state.error = "no zarr.json or .zattrs"
                return state
            }
            Object storeAttr = attrs[STORE_ATTR]
            if (storeAttr instanceof Map) {
                state.kind = storeAttr.kind
                state.schemaVersion = storeAttr.schema_version
            }
            Object labelling = attrs[LABELLING_ATTR]
            if (labelling instanceof Map) {
                state.status = labelling.status
                state.labellingSchemaVersion = labelling.schema_version
                Object landmarks = labelling.landmarks
                List pairs = (landmarks instanceof Map && landmarks.pairs instanceof List) ? (List) landmarks.pairs : []
                state.nPairs = pairs.size()
                state.nEnabledPairs = pairs.count { Object p -> !(p instanceof Map && p.enabled == false) }
            }
        } catch (Exception e) {
            state.error = "${e.getClass().getSimpleName()}: ${e.message}".toString()
        }
        return state
    }

    /** Errors that stop the register workflow before any task: store state, landmark counts, run ids. */
    static List<String> registerPreflight(List<Map> rows, Map params) {
        List<String> errors = []
        for (Map row : rows) {
            String id = row.id
            String p = row.prepared_store
            Map s = storeState(p)
            if (!s.exists) {
                errors.add(
                    "${id}: prepared store not found: ${p}; run workflows/manual_registration_prepare.nf with this samplesheet and --outdir first".toString()
                )
                continue
            }
            if (s.error) {
                errors.add("${id}: cannot read the root attributes of ${p}: ${s.error}".toString())
                continue
            }
            if (s.schemaVersion instanceof Number && s.schemaVersion > SCHEMA_VERSION) {
                errors.add("${id}: ${p} was written by a newer MerXen/viewer (schema_version ${s.schemaVersion} > 1); update".toString())
                continue
            }
            if (s.labellingSchemaVersion instanceof Number && s.labellingSchemaVersion > SCHEMA_VERSION) {
                errors.add(
                    "${id}: ${p} was written by a newer MerXen/viewer (schema_version ${s.labellingSchemaVersion} > 1); update".toString()
                )
                continue
            }
            if (s.kind == "result") {
                errors.add("${id}: ${p} is a result store, not a prepared store; set prepared_store to the ${id}_prepared.zarr store".toString())
                continue
            }
            if (s.kind != "prepared") {
                errors.add(
                    "${id}: ${p} is not a prepared manual-registration store (${STORE_ATTR}.kind = ${s.kind ?: 'missing'})".toString()
                )
                continue
            }
            if (s.status == null) {
                errors.add(
                    ("${id}: no landmarks saved in ${p}; open it in the viewer (Dataset loader > Manual Registration > " +
                    "Load image data), place the landmark pairs and press Finish labelling").toString()
                )
                continue
            }
            if (s.status == "draft") {
                errors.add("${id}: the landmarks in ${p} are a draft; open it in the viewer and press Finish labelling".toString())
                continue
            }
            if (s.status != "finished") {
                errors.add("${id}: unknown labelling status '${s.status}' in ${p}".toString())
                continue
            }
            for (String preset : (List<String>) row.presets) {
                int minPairs = effectiveMinPairs(row, preset, params)
                if (s.nEnabledPairs < minPairs) {
                    errors.add("${id} (preset ${preset}): ${s.nEnabledPairs} enabled landmark pairs in ${p}, at least ${minPairs} are needed".toString())
                }
            }
        }
        Map<String, Integer> runRows = [:]
        for (Map row : rows) {
            for (String preset : (List<String>) row.presets) {
                String runIdValue = runId(row.id.toString(), preset, row.run_label as String)
                if (runRows.containsKey(runIdValue)) {
                    errors.add(
                        "run id '${runIdValue}' is used twice (rows ${runRows[runIdValue]} and ${row.row_number}); set run_label to tell the runs apart".toString()
                    )
                } else {
                    runRows[runIdValue] = (Integer) row.row_number
                }
            }
        }
        return errors
    }

    /** One map per (row x preset), in row and preset order. */
    static List<Map> plannedRuns(List<Map> rows, Map params) {
        List<Map> runs = []
        for (Map row : rows) {
            for (String preset : (List<String>) row.presets) {
                String label = row.run_label as String
                runs.add([
                    run_id: runId(row.id.toString(), preset, label),
                    id: row.id,
                    preset: preset,
                    run_label: label,
                    run_dir: runDir(params, row.id.toString(), preset, label),
                    result_store: resultStorePath(params, row.id.toString(), preset, label),
                    summary_path: summaryPath(params, row.id.toString(), preset, label),
                    config_json: registerConfigJson(row, preset, params),
                ])
            }
        }
        return runs
    }

    // ------------------------------------------------------------------
    // Task configs (one map literal per marker; keys are checked against the pydantic models)
    // ------------------------------------------------------------------

    static String prepareConfigJson(Map row, Map params) {
        def referenceInput = [
            path: row.reference.path,
            scene: row.reference.scene,
            channel_names: row.reference.channel_names,
            registration_channel: row.reference.registration_channel,
            pixel_size_um: row.reference.pixel_size_um,
        ]
        def movingInput = [
            path: row.moving.path,
            scene: row.moving.scene,
            channel_names: row.moving.channel_names,
            registration_channel: row.moving.registration_channel,
            pixel_size_um: row.moving.pixel_size_um,
        ]
        def prepareConfig = [
            dataset_name: row.id,
            reference: referenceInput,
            moving: movingInput,
            prepared_dir: preparedDir(params, row.id.toString()),
            output_dir: "prepare_out",
            force: flag(params.manual_registration_force_prepare),
            workers: null,
        ]
        return JsonOutput.prettyPrint(JsonOutput.toJson(prepareConfig))
    }

    static String registerConfigJson(Map row, String preset, Map params) {
        def registerConfig = [
            dataset_name: row.id,
            preset: preset,
            run_label: row.run_label,
            prepared_store_path: row.prepared_store,
            result_dir: runDir(params, row.id.toString(), preset, row.run_label as String),
            output_dir: "register_out",
            global_settings: (params.manual_registration_settings ?: "").toString().trim(),
            row_settings: row.registration_settings ?: "",
            force: flag(params.manual_registration_force_register),
        ]
        return JsonOutput.prettyPrint(JsonOutput.toJson(registerConfig))
    }

    static String viewerCacheConfigJson(Map row, Map params) {
        def viewerCacheConfig = [
            dataset_name: "${row.id}_prepared".toString(),
            latest_zarr_path: "latest_input.zarr",
            output_dir: "viewer_cache_out",
            image_only: true,
            image_keys: ["reference", "moving"],
            downsample: params.viewer_cache_downsample,
            min_size: params.viewer_cache_min_size,
            build_image_pyramid: true,
        ]
        // A param that is absent leaves the model default in force.
        Map present = viewerCacheConfig.findAll { String k, Object v -> v != null }
        return JsonOutput.prettyPrint(JsonOutput.toJson(present))
    }

    static String compareConfigJson(String id, List<String> summaryPaths, Map params) {
        def compareConfig = [
            dataset_name: id,
            summary_paths: summaryPaths,
            output_dir: "${outdirAbs(params)}/${id}/registered".toString(),
            task_output_dir: "compare_out",
        ]
        return JsonOutput.prettyPrint(JsonOutput.toJson(compareConfig))
    }
}
