# Changelog

## 0.2.0 (unreleased; gate H of the robust cell-type annotation plan)

### Changed: human clustering is map-first by default

- Human runs now label cells with MapMyCells against the packaged reference bundles (WHB frontal cortex, with SEA-AD as the second vote), then build the hierarchy from those labels (`clustering_squidpy_mode_human = "map_first"`). This replaces the legacy Leiden-then-label flow. Mouse stays `legacy` until its own flip (M9).
- **Tables.** A human `map_first` run writes the unsuffixed clustered tables (`table_<segmentation>_clustering_squidpy`), so a rerun into an existing results directory replaces the legacy tables. Keep a snapshot first, or set `--clustering_squidpy_table_key_suffix mapfirst`.
- **Legacy MAPMYCELLS stage.** It no longer runs beside map-first for a flipped species. `annotation_mode_mapmycells_stage` now defaults to `skip` for human and `legacy` for mouse.
- **Inputs.** Human runs need the reference-source and reference-store params; preflight checks them. See [configuration](docs/configuration.md) and [reference-based annotation](docs/stages/annotation.md).
- **Legacy mode.** Human `legacy` still runs, but it is deprecated and logs a warning.

### Added

- **Per-row mode.** A `clustering_squidpy_mode` samplesheet column (`legacy` / `map_first`; blank follows the run), so one run can opt single datasets in or out. A header that only resembles the column is reported. In a run with any `map_first` row, legacy rows also take the MENDER barrier's `combine`, so they run MENDER where a legacy-only run's `join` would skip it.
- **Label columns.** Per-level labels with status, bootstrap probability and method agreement (`ct_<level>_name`, `ct_<level>_status`, `ct_<level>_raw`, `ct_consensus_tier`, …), plus the map-first hierarchy keys `ct_branch`, `ct_leaf` and `ct_mender_state`. See [outputs](docs/outputs.md).
- **Statuses and broad classes.**
  - A label can be `not_resolvable`: resolvability, a self-map of the reference on simulated cells, does not emit that level, class and depth bin for the dataset's composition.
  - New broad values: `Mixed/Unknown` (no confident level) and `Oligodendrocyte lineage` (OPC vs COP unresolved).
- **Panel trust states:** `refused`, `broad_only`, `validated` and `provisional`, per reference and panel, with banners and dataset-gate warnings.
- **Hierarchy.** `broad_class` and `hierarchical_cluster` now come from the mapped labels: mapped reference types are the leaves (`leaf_source = mapped`).
- **MENDER.** Map-first tables use the `exclude_from_features` policy: unassigned cells stay spatial nodes but add no neighbourhood state. Legacy tables keep `state`.
- **Annotation QC report** per pair and segmentation (`annotation_report/`).
- **Cortical depth.** A `boundary_frame` setting computes depth in the frame of the manual boundaries (the MERSCOPE frame fix).
- **New panels in RESOLVE.** Panels outside the validated families use resolvability version 7 (custom and unknown panels, Xenium Prime 5K), and RESOLVE now reads those bundles instead of stopping: it applies the version-7 ensemble decisions, writes the report-only label column `flag_nonneuronal_high_depth` (null for version-6 panels) and records the class-depth prediction in the resolve summary. Reading version 7 leaves version-6 outputs unchanged apart from that null column; the panel record and mouse gate changes below apply to every bundle version.
- **Panel record and the mouse 10% rule.**
  - The panel record of the provenance (`panel`) is now built the same way for mouse as for human, with the panel family, family basis, validation basis, validated level and table digests, panel mode, validated shares and gene-ID diagnostics. Mouse records used to hold only the panel hash, trust state, reasons, banner and missing-gene count.
  - Without panel diagnostics (no primary run, no panel file, or diagnostics that do not fit the bundle), human and mouse records still carry the trust decision's fields, the panel mode and the validated shares; only the gene-ID fields stay empty. Human records used to leave all of these empty in that case.
  - A mouse panel family validated by simulation now warns (`unvalidated_share:<level>`) when more than `warn_unvalidated_share` (10%) of the confident labels at broad, class, nt or subclass fall outside its validated region, as the human gate does. The warning never changes the gate level.

### Acceptance

- The human acceptance (gate H) is scored in `results/acceptance/2026-09-30/` on P7513, P1212, P7113 and P5011, all four segmentations:
  - `summary.html`: stage B;
  - `rescore_2026-10-01/`: the user's decisions of 2026-10-01;
  - `segmentation_comparison/`: reseg vs proseg_hybrid.
- The protocol and the user's decisions are in `docs/acceptance/annotation-v1-preregistration.md` §18–§20.
