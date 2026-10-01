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

### Acceptance

- The human acceptance (gate H) is scored in `results/acceptance/2026-09-30/` on P7513, P1212, P7113 and P5011, all four segmentations:
  - `summary.html`: stage B;
  - `rescore_2026-10-01/`: the user's decisions of 2026-10-01;
  - `segmentation_comparison/`: reseg vs proseg_hybrid.
- The protocol and the user's decisions are in `docs/acceptance/annotation-v1-preregistration.md` §18–§20.
