/**
 * Registry of the supported platforms, twin of src/merxen/platforms.py.
 *
 * Each entry holds the names and samplesheet fields the workflow used to spell
 * out in binary MERSCOPE/XENIUM branches. forName() raises for a platform that
 * is not registered instead of letting it fall into another platform's branch.
 * Adding a platform means adding an entry here and in platforms.py
 * (tests/test_workflows/test_platform_registry_groovy.py checks they agree).
 */
class PlatformSpec {
    final String name
    final String family
    final String nativeImageKey
    final String originalCellShapeKey
    final String originalNucleusShapeKey
    final List<String> defaultChannels
    final List<String> inputDirFields
    final List<String> spatialdataPathFields
    final List<String> transformPathFields

    static final Map<String, PlatformSpec> REGISTRY = Collections.unmodifiableMap([
        MERSCOPE: new PlatformSpec(
            'MERSCOPE',
            'MERSCOPE',
            'MERSCOPE_z_projection',
            'merscope_cell_boundaries',
            null,
            ['DAPI', 'PolyT'],
            ['merscope_dir'],
            ['merscope_spatialdata_path', 'merscope_zarr_path'],
            ['merscope_transform_path'],
        ),
        XENIUM: new PlatformSpec(
            'XENIUM',
            '10X',
            'morphology_focus',
            'xenium_cell_boundaries',
            'xenium_nucleus',
            ['DAPI', '18S'],
            ['xenium_dir'],
            ['xenium_spatialdata_path'],
            ['xenium_spec_path'],
        ),
    ] as LinkedHashMap<String, PlatformSpec>)

    PlatformSpec(
        String name,
        String family,
        String nativeImageKey,
        String originalCellShapeKey,
        String originalNucleusShapeKey,
        List<String> defaultChannels,
        List<String> inputDirFields,
        List<String> spatialdataPathFields,
        List<String> transformPathFields
    ) {
        this.name = name
        this.family = family
        this.nativeImageKey = nativeImageKey
        this.originalCellShapeKey = originalCellShapeKey
        this.originalNucleusShapeKey = originalNucleusShapeKey
        this.defaultChannels = Collections.unmodifiableList(new ArrayList<String>(defaultChannels))
        this.inputDirFields = Collections.unmodifiableList(new ArrayList<String>(inputDirFields))
        this.spatialdataPathFields = Collections.unmodifiableList(
            new ArrayList<String>(spatialdataPathFields))
        this.transformPathFields = Collections.unmodifiableList(
            new ArrayList<String>(transformPathFields))
    }

    /**
     * Return the entry of a platform.
     *
     * The name must match exactly (as the former == comparisons did); a GString
     * is looked up by its String value.
     *
     * @throws IllegalArgumentException if the platform is not registered
     */
    static PlatformSpec forName(platform) {
        def spec = REGISTRY.get(platform?.toString())
        if (spec == null) {
            throw new IllegalArgumentException(
                "Unknown platform: ${platform}. Supported: ${REGISTRY.keySet().join(', ')}"
            )
        }
        return spec
    }

    /** Names of the registered platforms, in registry order. */
    static List<String> supportedPlatforms() {
        return new ArrayList<String>(REGISTRY.keySet())
    }

    /** A fresh, mutable copy, so callers never alias the registry list. */
    List<String> getDefaultChannels() {
        return new ArrayList<String>(defaultChannels)
    }

    /** Plain map of every field (the Python parity test reads it). */
    Map toMap() {
        return [
            name: name,
            family: family,
            native_image_key: nativeImageKey,
            original_cell_shape_key: originalCellShapeKey,
            original_nucleus_shape_key: originalNucleusShapeKey,
            default_channels: getDefaultChannels(),
            input_dir_fields: new ArrayList<String>(inputDirFields),
            spatialdata_path_fields: new ArrayList<String>(spatialdataPathFields),
            transform_path_fields: new ArrayList<String>(transformPathFields),
        ]
    }
}
