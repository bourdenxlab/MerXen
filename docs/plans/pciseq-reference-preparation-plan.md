# pciSeq reference-preparation plan

Branch: `feature/pciSeq`. Extends the optional typing integration described in
[pciSeq integration](pciseq-integration-plan.md).

## Decisions

- Keep native polygon-based 2D seeds and original measured transcript
  coordinates. Prepared raster masks remain disposable work-directory data.
- Default to Allen WMB for mouse and healthy WHB for human; offer SEA-AD and
  custom mean-expression matrices explicitly.
- Require human anatomical regions. Use whole brain for mouse until future
  Allen CCFv3 coronal-section registration can supply covered regions plus a
  100 µm safety margin. Preserve explicit manual mouse-region selection.
- Use broad subclass labels by default. WHB has no subclass level, so use its
  intermediate cluster level and retain native hierarchy names.
- Use all 84 SEA-AD donors by default, averaging donors equally within each
  type. Keep explicitly defined control and AD cohorts selectable.
- Build linear normalized arithmetic means from raw expression, not marker
  lookup tables, log expression, or MapMyCells precomputed statistics.

## Delivery

1. Inspect local reference/config inventories and released Allen schemas.
   Verify legacy files against official hashes or their recorded bundle hashes.
   Recover healthy contents to the configured reference store and reacquire
   damaged official data without deleting original sources. Keep a private
   audit record of paths, checksums, failures, and recovery outcomes.
2. Add pinned-manifest acquisition with checksum/structure verification,
   resumable downloads, and locks. Reuse existing healthy atlas files per file.
3. Join region, donor, and taxonomy tables using native identifiers. Validate
   requested levels and regions, and report unmatched cells explicitly.
4. Stream raw matrices, normalize across original genes, aggregate by donor and
   type, enforce support thresholds, and publish reproducible cached bundles.
5. Add configuration and CLI support. Share prepared references across
   Nextflow branches, propagate provenance to typing outputs, and support
   staged custom matrices with optional class/region metadata.
6. Validate known-value fixtures, malformed inputs, cohort/region filters,
   content-cache repair, source downloads, actual atlas preparation, and a
   complete small Nextflow typing run. Run repository checks before committing.

Operational usage and dataset details are in
[pciSeq reference preparation](../stages/pciseq-references.md).

## Validation

- Full repository checks pass: 1,865 tests passed, 35 skipped, Ruff lint/format,
  mypy, and the metro-map validator.
- Known-value fixtures cover the three native atlas joins, regional/cohort
  selection, donor weighting, full-library normalization, duplicate gene
  symbols, unavailable expression, malformed counts, and prepared-cache repair.
- Asset tests cover manifest pinning, hash failures, interrupted/resumed
  downloads, read-only existing caches, and damaged verification sidecars.
- Actual WMB preparation verifies all 13 v3 shards and produces a small-panel
  cerebellar test reference. Actual WHB preparation verifies both raw matrices
  and produces a small-panel cortical reference. Actual SEA-AD preparation
  validates released metadata joins and a regional small-panel reference.
- A complete small Nextflow run executes the real upstream fit, validates
  custom matrix/metadata staging and ROI restriction, preserves large canonical
  IDs and clustered counts, and publishes provenance plus a derived table.

Private input inventories, complete checksums, recovery failures, and actual-data
validation outputs are kept outside the repository under
`$MERXEN_REFERENCES_DIR/pciseq_audit`.
