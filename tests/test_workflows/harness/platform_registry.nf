// Appended by test_platform_registry_groovy.py to the Groovy functions of
// workflows/main.nf and the span wrappers of the golden harness. It prints
//
//     REGISTRY <TAB> <compact JSON of PlatformSpec.REGISTRY>
//
// for the parity test with src/merxen/platforms.py.
//
// Nextflow strict syntax: only top-level functions may be called by name.

def printRegistry() {
    def registry = PlatformSpec.REGISTRY.collectEntries { name, spec -> [name, spec.toMap()] }
    println "REGISTRY\t" + groovy.json.JsonOutput.toJson(registry)
}

workflow {
    printRegistry()
    println "DONE"
}
