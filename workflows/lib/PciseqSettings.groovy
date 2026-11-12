/** pciSeq compute-mode settings, checked before any pciSeq task starts. */
class PciseqSettings {
    static final List<String> MEMORY_MODES = ['standard', 'batched']
    static final List<String> BATCH_SIZE_PARAMS = [
        'pciseq_cell_batch_size', 'pciseq_spot_batch_size',
    ]

    /** True when the experimental disk-backed compute process is selected. */
    static boolean batched(params) {
        return params.pciseq_memory_mode?.toString() == 'batched'
    }

    /** Errors for the memory mode and batch sizes; empty when they are valid. */
    static List<String> errors(params) {
        def errors = []
        def mode = params.pciseq_memory_mode
        if (mode == null || !(mode.toString() in MEMORY_MODES)) {
            errors << "pciseq_memory_mode must be standard or batched, not '${mode}'".toString()
        }
        BATCH_SIZE_PARAMS.each { name ->
            if (!positiveInteger(params[name])) {
                errors << "${name} must be a positive integer, not '${params[name]}'".toString()
            }
        }
        return errors
    }

    static boolean positiveInteger(value) {
        if (value instanceof Integer || value instanceof Long) {
            return value >= 1
        }
        return value instanceof CharSequence && value.toString() ==~ /[1-9][0-9]{0,9}/
    }
}
