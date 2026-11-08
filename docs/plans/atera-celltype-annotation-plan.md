# AT-11: whole-transcriptome human hippocampal annotation

This work targets `feature/atera-support`. It does not enable Atera map-first
annotation by borrowing the validated frontal-cortex settings or family trust.
The existing gates remain in place until the new reference and validation are
ready. Experimental calls are written separately from published annotations.

## Fixed requirements

- Use all 18,049 measured panel genes in each spatial query. Never trim the
  query to its marker genes: raw-query CPM depends on its full row sums.
- Preserve the existing stores and segmentation outputs. Keep unexpressed
  objects explicit rather than forcing a cell-type call.
- Treat hippocampus as its own reference domain. Compare a hippocampal
  reference with one including the adjacent parahippocampal dissections.
- Prefer measured accuracy to inherited marker limits, family promotion,
  confidence thresholds or depth caps. More markers alone do not prove greater
  accuracy; compare settings on exactly the same held-out cells and depths.
- Keep existing Xenium/MERSCOPE annotation behavior unchanged.

## Execution order

1. **Reference and validation audit.** Recover the measured reference probe,
   verify gene IDs and regional coverage, and identify normalization and
   train/test leakage risks. The recovered all-donor hippocampal precompute is
   useful for feasibility, but cannot provide donor-held-out accuracy.
2. **Donor-separated development experiment.** Exclude one atlas donor before
   computing reference statistics or markers. Train on every eligible cell of
   the remaining donors, with the minimum leaf support recorded. Evaluate a
   deterministic, class-balanced held-out sample; retain truth labels absent
   from training in the accuracy denominator. Test native and spatial-like
   transcript depths using all panel columns.
3. **Marker/reference comparison.** Compare the two regional scopes and the
   current marker selection with expanded selections. Report overall accuracy,
   macro F1 over truth-present classes, per-class support, unseen truth and
   precision/coverage versus bootstrap support. Bootstrap support is not assumed
   to be a calibrated correctness probability. Freeze the development choice
   before confirmation on the other donors.
4. **Independent confirmation and spatial pilot.** Confirm the frozen choice
   on the other held-out donors, including rare classes and classes absent from
   training. Build the final reference from all eligible donors and map vendor,
   Cellpose and Cellpose-prior ProSeg queries into new output directories.
   Inspect marker agreement, anatomical distributions and changes from the
   provisional legacy labels. Vendor annotations are an audit comparison, not
   ground truth.
5. **Production integration.** Introduce the measured reference/profile and
   independently justified confidence/depth rules into the unpaired Atera RCA
   path. Lift the workflow gate only with tests covering mixed samplesheets,
   preserved small-panel behavior and the complete annotation report. Any
   unresolved or unsupported class must remain visible in the report.

## Evidence and acceptance

Private locations and identifiers live under
`$MERXEN_EVIDENCE_ROOT/atera_at11_probe` and the AT-11 continuation evidence;
source locations come from the local configuration. Keep immutable source
snapshots, configurations, donor/cell exclusion audits, reference statistics,
marker lookups, mapping outputs, timing and memory records.

No production confidence gate is inherited merely because it passed on a
targeted frontal panel. Low support, missing reference classes, unavailable
reference genes, empty objects and disagreement must be reported explicitly.
Final fine labels require held-out evidence at relevant depths; broad labels
and unassigned outcomes are valid when finer distinctions lack that evidence.

## Hippocampal policy implementation

`annotation.hippocampal_profile` provides reference-bound policy fitting and
resolution independently of the frontal annotation defaults. A binding records
species, domain, regional scope, ordered taxonomy, the full-panel identity and
hashes of reference statistics, marker lookup and tree. A runtime query must
preserve all 18,049 ordered columns. Artifact changes invalidate the binding.

The operating objective is explicit in each policy, rather than inherited from
an existing panel family. Class/depth rules maximize observed accepted-cell
coverage subject to that objective; repeated thinnings count as one cell per
class/depth interval, choosing the lowest-depth realization before examining
correctness. Unsupported classes and depths remain unresolved. Raw predictions
survive abstention, and a child cannot be emitted without its matching parent.

The current evidence basis is `existing_atlas_diagnostics_reused`. Additional
donor collection is not an acceptance prerequisite for this dataset. Policies
fitted or inspected using the existing atlas folds must not be described as a
new independently confirmed policy, or as spatial ground-truth validation.
Per-class precision, coverage, uncertainty and all unresolved outcomes remain
part of the acceptance report.

This module does not lift the Atera workflow gate. Configuration dispatch,
pipeline/report integration and mixed-samplesheet plus crop/full-section exit
checks must pass before activation. Existing viewer review exports remain
immutable snapshots of the experimental calls.
