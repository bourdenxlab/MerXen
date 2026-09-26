/*
 * Reference-based cell-type annotation processes (plan §3.2, §3.6, §11.4).
 *
 * Milestone M1 (contract scaffolding) defines no process yet:
 * ANNOTATE_PANEL and ANNOTATE_REFERENCE_PREP arrive in M2 and
 * ANNOTATION_REPORT in M7. Hook H1 in main.nf already includes this module,
 * so nothing here may run while both species use the legacy clustering mode.
 */

// Placeholder that lets hook H1 include this module before it has a process.
// M2 replaces it in the H1 include with the first annotation processes.
def annotationModuleStub() {
    return "annotation module scaffolding (M1): no processes"
}
