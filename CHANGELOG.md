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

### Added: new-panel onboarding and simulation-based validation (M13)

- **Gate P.** `merxen annotation-panel-simulate --gate-p` and `scripts/acceptance/new_panel.py` validate a new panel family by simulation (plan §14, NP1–NP9). They write the NP1–NP9 report and per-(level, class) records under `<out-dir>/gate_p/`. See [gate P](docs/stages/annotation.md#gate-p-simulation-based-validation-of-a-panel-family-m13).
  - Leave-one-donor-out held-out bundles for the two other frontal WHB donors, each drawing its own other-region test cells, built in a separate gate-P store. The production store is refused.
  - Mapping seeds 0 and 1, NP6 stress members (spill, LogNormal efficiency, cross-platform offsets, the lung stress for human Prime), the clean upper bound and NP9's identity re-runs.
  - `--species` is required, and only the human path is implemented. A mouse family is refused until the second WMB test draw exists (after M6b).
  - No family has been promoted. The set a dry run runs first, and the readings put to the user in pre-registration §23.11–§23.17 are open until the user rules on them.
  - `simulate_report.json` and `gate_p_run.json` record the code that ran (`code_commit`, `code_commit_source`). A `git archive` export has no `.git`, so the command reads the commit from a `COMMIT` file at the export's root, refusing a malformed one before any compute; a worktree is asked with git.
- **Promotion writer.** `diagnostics.write_simulation_family` adds a passing family's rows in its gate-P PR: the `validated_panels.csv` row (`validation_basis = simulation`), its `validated_panel_levels.csv` rows and its `validated_panel_genes.csv` rows. It refuses a family without a self-map, a `family_id` other than the panel's own, and rows for classes or levels RESOLVE does not read.
- **Cross-platform stress asset.** `ratio__xenium_v1_vs_merscope__human_brain_ffpe` (set a's measured Xenium / MERSCOPE offsets, with a keyed resample for the genes it does not cover, capped at ±2 log2) is registered as an in-house simulation input with provenance. The `sim_inputs` NOTICE now lists in-house assets.
- **Real-data QC in human RESOLVE** (plan §8.8). Every human sample now runs the downgrade-only QC:
  - checks: the marker referee, registration G1, paired concordance, flag rates, gene complexity, per-class coverage against simulation, the prefilter spot check and the factor re-measure;
  - outcomes: `pass`, `warn`, `fail`, `not_applicable` or `not_evaluable`, recorded in the provenance (`panel.real_data_qc`), the resolve summary (`real_qc`; `schema_version` 3) and the report's panel card (the `panel_real_qc` table);
  - effects: a check can only lower a dataset (cap its gate level, withhold a level, withhold the pair's supercluster-level cross-platform statistics). It never raises a trust state or changes a margin;
  - the seeded real-data families only warn until their species gate merges into `main`. `real_qc.seeded_families_warn_only_until_gate` is now a per-species record; the old bool still loads;
  - `real_qc.enabled = false` gives the QC-free run.
  See [configuration](docs/configuration.md) for the `real_qc` fields.
- **Registration checks reach RESOLVE.** In `map_first` runs the RESOLVE task now receives the QC stage's registration checks (`*_registration_qc.json`), so it waits for the QC stage of its pair and segmentation. Human RESOLVE uses them for registration G1, warn-only in M13.
- **Simulated genes in version-7 PREP.** A version-7 self-map also writes `resolvability_sim_genes.parquet`, each simulated cell's realised total and detected genes, for the gene-complexity check. Its version enters the version-7 `build_hash`, so version-7 bundles built before it are rebuilt into new build directories rather than reused (they stay readable, with the check `not_evaluable`). Version-6 bundles and their build hashes are unchanged.
- **Regression script.** `scripts/acceptance/m13_real_qc_regression.py` tabulates the real-data QC outcomes on the M8 human pairs.
- **The user's rulings of 2026-10-07** (pre-registration §23.19) change these defaults:
  - the human marker referee derives its marker sets with the `class` comparator (`real_qc.marker_referee_comparator`); the 0.75 / 0.70 thresholds are unchanged. The `node` comparator, the default until then, was `not_evaluable` on every set a sample. The regression script pins the settings C17 ran with, so a re-run reproduces C17.

### Fixed

- **Trust fail-safe of simulation families.** A panel family validated by simulation whose bundle has no self-map now stays `broad_only`, the provisional fail-safe, with the family verdict as a note. Before, it became `validated`, so promoting such a family would have changed what RESOLVE emits. Families validated on real data keep their state. No simulation family is listed yet, so no output changes.

### Acceptance

- The human acceptance (gate H) is scored in `results/acceptance/2026-09-30/` on P7513, P1212, P7113 and P5011, all four segmentations:
  - `summary.html`: stage B;
  - `rescore_2026-10-01/`: the user's decisions of 2026-10-01;
  - `segmentation_comparison/`: reseg vs proseg_hybrid.
- The protocol and the user's decisions are in `docs/acceptance/annotation-v1-preregistration.md` §18–§20.
