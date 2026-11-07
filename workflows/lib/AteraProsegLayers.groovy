/**
 * The ProSeg layer of each prior of the tiled ATERA ProSeg route, twin of
 * PROSEG_LAYERS in src/merxen/segmentation/atera_proseg/__init__.py
 * (tests/test_workflows/test_platform_registry_groovy.py checks they agree).
 *
 * native_proseg rows (AT-9) refine the vendor cell masks: layer proseg_vendor.
 * cellpose_proseg ATERA rows (AT-7) refine MerXen's Cellpose masks: layer reseg
 * with the XENIUM element names.
 */
class AteraProsegLayers {
    static final Map<String, Map<String, Object>> REGISTRY = Collections.unmodifiableMap([
        vendor: Collections.unmodifiableMap([
            prior_kind: 'vendor',
            analysis_segmentation: 'proseg_vendor',
            branch: 'proseg_vendor',
            shapes_key: 'MOSAIK_proseg_vendor',
            table_key: 'table_MOSAIK_proseg_vendor',
            label_key: 'proseg_vendor_cell_labels',
            id_namespace: 'source_segmentation_v1',
            prior_name: 'vendor_cell_labels',
            legacy_aliases: ['vendor_proseg', 'native_proseg'],
        ]),
        cellpose: Collections.unmodifiableMap([
            prior_kind: 'cellpose',
            analysis_segmentation: 'reseg',
            branch: 'proseg',
            shapes_key: 'MOSAIK_proseg',
            table_key: 'table_MOSAIK_proseg',
            label_key: 'proseg_cell_labels',
            id_namespace: 'cellpose_label_v1',
            prior_name: 'cellpose_cell_labels',
            legacy_aliases: ['reseg'],
        ]),
    ] as LinkedHashMap<String, Map<String, Object>>)

    static final Map<String, String> SOURCE_PRIOR_KIND = Collections.unmodifiableMap([
        native_proseg: 'vendor',
        cellpose_proseg: 'cellpose',
    ] as LinkedHashMap<String, String>)

    /** The layer of a prior kind; throws for an unknown kind. */
    static Map<String, Object> forKind(kind) {
        def layer = REGISTRY.get(kind?.toString())
        if (layer == null) {
            throw new IllegalArgumentException(
                "Unknown ATERA ProSeg prior kind: ${kind}. Known: ${REGISTRY.keySet().join(', ')}"
            )
        }
        return layer
    }

    /** The prior kind of a segmentation source, or null when it runs no tiled ProSeg. */
    static String priorKindForSource(source) {
        return SOURCE_PRIOR_KIND.get(source?.toString())
    }
}
