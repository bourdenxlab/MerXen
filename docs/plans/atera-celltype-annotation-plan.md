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

## Explicit whole-panel workflow

The whole-panel path is separate from the frontal `AnnotationConfig` and label
schema. `--annotation_hippocampal_resources` selects a local resource manifest;
ATERA rows must declare human `anatomical_region=hippocampus`. Without this
manifest ATERA keeps legacy mode, including its existing task hashes. Xenium
and MERSCOPE continue through the existing annotation workflow.

The resource stage verifies the measured panel, statistics, lookup, taxonomy,
frozen policy and qualification identities. Gene-release replacements replay
the panel's recorded resolver and must reproduce its feature-resolution digest.
All 18,049 query columns and every raw count survive; empty objects are explicit.
The measured mapping settings are 100 bootstraps, factor 0.5, seed 0, eight
workers, chunks of 1,000 and `max_gb=40`. These are independent of frontal caps.

`PolicyQualification` checks each emitting class/depth rule against at least two
existing diagnostic folds. A failed or unsupported check withholds the rule;
it does not retune its bootstrap cut. Qualified children still require qualified,
accepted parents. SEA-AD Class → Subclass → Supertype provides final labels,
falling back to an accepted ancestor. Allen WHB Supercluster → Cluster remains
a separate comparison, never a fabricated cross-taxonomy consensus.

The workflow separates resource verification, MAP, RESOLVE, COMPUTE and REPORT.
MAP hashes the actual prepared H5AD files and mapping identities. A changed
policy or qualification identity invalidates RESOLVE without invalidating MAP.
COMPUTE writes the hierarchy, a CPU QC embedding, the raw-count layer and profile
provenance. Its table suffix is `rcahippocampal`, preserving legacy outputs.
Unassigned objects remain spatial nodes but add no MENDER cell-state feature.
Preparation, finalization and MENDER read only the selected table and native
shapes for this profile, so a sibling branch writing a derived table cannot
interrupt preparation. Existing legacy and Xenium/MERSCOPE reads remain unchanged.

Reports retain all resolution reasons, full-denominator coverage by depth,
class/depth diagnostic support and uncertainty, separate atlas calls, subclass
UMAP/spatial figures and the MENDER completion record when requested. Atlas
agreement is not spatial accuracy. MENDER cross-platform comparability is not
inherited from the frontal profile. The report fails if requested MENDER outputs
are absent. Registration, spatial ground truth and newly independent donor
confirmation are not implied by successful workflow execution.

Acceptance evidence must cover the mixed-profile report/restart harness, the
existing ATERA and frontal regressions, and real crop/full-section annotation
through MENDER on isolated stores. Completed review exports are immutable;
updated annotations require a new export version.
