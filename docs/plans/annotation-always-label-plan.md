# Always label: dataset gates become metadata

> **Status:** approved 2026-11-25 with changes to O3 and O6 (see Decisions). M0 built; M1-M3
> and M5 in progress on one branch; M4 deferred (O5).

Today map-first annotation can withhold labels from a whole dataset (one sample × segmentation).
The dataset gate's level (`full`, `broad_only`, `failed`), the panel's trust state (`refused`,
`broad_only`) and some real-data QC effects (`gate_cap`) turn computed labels into
`not_attempted_gate` with no name, set `exclude_hard` on every cell, and make the downstream
steps (hierarchy leaves, MENDER, cross-platform statistics, report plots) skip or degrade.

A sparse new mouse MERSCOPE section (about 3 M transcripts, median 15 per cell, no alias yet)
showed the cost: three of its four segmentations failed the mouse gate on G1's density ratio
(1.24-1.39 < 1.5) with a measured shift of 0 µm, and lost every label, although MapMyCells had
mapped every cell. The one segmentation that passed was the vendor's over-split 3-D
segmentation with a median of 1 transcript per cell.

The owner's decision (2026-11-24): labels are always produced, for every sample, segmentation
and panel, whatever the panel, transcript quality or other metrics. What the gates detect is
written into each cell's metadata, so a user filters later to the cells and the label level
they trust.

## Principles

| # | Principle |
|---|---|
| P1 | No dataset-level or panel-level condition removes a label. A cell gets a label at every level the engine called it. Only the cell's own data can leave it without one (it was not mapped: off-table, or no usable gene). |
| P2 | `ct_<L>_status` describes the cell's own call at that level: `confident`, `low_confidence`, `below_floor`, `not_resolvable`, `implausible`, `parent_unresolved`, `single_method`, `method_disagree`, `low_counts`. `not_attempted_gate` is retired. |
| P3 | Every dataset-level signal (gate signals G1-G5 and the human gate's depth and coverage rules, panel trust, real-data QC outcomes) is written per cell as columns that are constant within the dataset, and stays in the resolve summary. |
| P4 | Where a gate signal has a per-cell counterpart, the cell carries it too (marker referee agreement, transcript enrichment over local background, spill-over, implausibility). |
| P5 | Today's conservative answer is kept as derived `*_strict` columns, so nothing a user relies on is lost and the acceptance scripts keep their meaning. One code path computes both. |
| P6 | Downstream steps run on every dataset and record the dataset's QC state next to their results instead of skipping. |

## Per-cell columns (label table, and `obs` of the clustered table)

Kept, with new meaning:

- `ct_<L>_name`, `_raw`, `_conf`, `_corr`, `_runner_up`, `_margin`: filled whenever the engine
  called the cell at level L (today blanked under a failed gate, under a `broad_only` gate for
  the leaf levels, and for a refused primary).
- `ct_<L>_status`: the cell's own status (P2).
- `ct_final_level` / `ct_final_name`: the deepest level whose status is `confident`, by the
  cell's own statuses only.
- `exclude_hard`: cell-intrinsic only (off-table, or implausible at every level it was called).
  No longer set by the dataset gate.
- Existing per-cell flags stay: `flag_low_counts`, `flag_below_floor`, `flag_contaminated`,
  `flag_diffuse_profile`, `flag_ood`, `flag_microglial_spillover`, `flag_region_incoherent`,
  `flag_implausible`, `flag_astro_lowcount`, `flag_method_disagree`, `discovery_caution`.

New, dataset-level, constant within a dataset (P3):

| Column | Content |
|---|---|
| `qc_dataset_level` | What the gate would have enforced: `full`, `broad_only`, `failed`. Advisory. |
| `qc_dataset_max_level` | The deepest chain level the dataset QC supports: the leaf when the gate is `full`, the level above it (`nt`) when `broad_only`, above a level real-data QC withholds, `none` when the gate failed or the primary is refused or missing. |
| `qc_dataset_warnings` | `;`-joined tokens of every signal that warned or tripped a fail rule, e.g. `g1_density_ratio_fail`, `g2_marker_consistency_warn`, `g4_astro_epen`, `depth_frac_ge30_low`, `table_broad_coverage_low`, `panel_provisional`, `qc_marker_consistency_cap`. Empty when clean. |
| `qc_panel_trust` | `validated_real`, `validated_simulation`, `provisional`, `broad_only`, `refused`. |
| `qc_g1_density_ratio`, `qc_g2_marker_consistency`, … | The numeric signals, so a user can apply their own cut-off. |

New, per cell (P4; milestone M4):

| Column | Content |
|---|---|
| `cell_transcript_enrichment` | Transcript density inside the cell over the local background density (transcripts outside cells within ~50 µm). G1's question asked per cell: does this cell stand out from the background? |
| `flag_low_enrichment` | `cell_transcript_enrichment` below a configured value. |
| `marker_referee_group`, `flag_marker_referee_disagree` | G2 per cell: the marker pseudo-label (where the cell has one) and whether its class call disagrees. |

Derived strict view (P5):

- `ct_final_level_strict` / `ct_final_name_strict`: `ct_final_*` capped at
  `qc_dataset_max_level` (`none` when the dataset would have failed). Equals today's
  `ct_final_*`.
- `ct_<L>_status_strict`: `ct_<L>_status`, or `not_attempted_gate` where today's gate or trust
  would have blocked level L.
- `exclude_hard_strict`: `exclude_hard` or `qc_dataset_level == failed`.

A user filters with plain pandas, e.g. confident subclass labels in datasets with no fail-rule
signal:

```python
keep = (obs.ct_subclass_status == "confident") & ~obs.qc_dataset_warnings.str.contains("_fail")
```

M5 adds a small helper, `merxen.annotation.filtering.select_cells(obs, level=..., cell=...,
dataset=...)`, and a docs page of such recipes.

## What changes, by area

### M0: done 2026-11-24 (on `main`, uncommitted at the time of writing)

- G1's fail rule only warns by default: new `mouse_gate.g1_effect = warning`, and the human
  `real_qc.registration_g1_effect` default is `warning` again.
- The launch log no longer lists a skipped `mapmycells` stage, and warns a row that stops there.

### M1: RESOLVE emits every label (core)

- `consensus.py`: drop the `blocked` loops (human ~`:1506`, mouse ~`:2120`) and the `leaf_gated`
  branches (human `:1416,1444,1477`; mouse `:2050,2059,2096`), so every level is resolved from
  the cell's own evidence. Remove `not_attempted_gate` from `NAMELESS_STATUSES` and the status
  vocabulary (`schema.py:67-85`); `_deepest_and_exclude` (`:1598-1625`) stops setting
  `exclude_hard` from the gate.
- `thresholds.dataset_gate`, `mouse_gate.evaluate_mouse_gate`: keep the signals and the level,
  but the level becomes advisory (`qc_dataset_level`); `TrustDecision.gate_level_cap` and
  `exclude_hard` (`diagnostics.py:1724-1811`) feed only the advisory level and the strict view.
- `real_qc.py`: `gate_cap` and `withhold_level` effects become warning tokens; `apply_qc_to_gate`
  writes the advisory level only; the human re-run with QC applied (`pipeline.py:5085`) goes.
  `withhold_pair_stats` becomes a recorded restriction (M3).
- The `g1_effect`, `registration_g1_effect` and `marker_consistency_broad_only` settings then
  only change the advisory level. They stay, documented as advisory.
- Label table: add the P3 and P5 columns (`schema.py:212-265`, `column_specs`), bump the
  label-table and resolve-summary schema versions; the summary keeps `mouse_gate`, `trust`,
  `banner`, `real_qc`, with the level renamed in prose to "advisory".
- Tests: about 70 tests pin the blocked behaviour (consensus, mouse gate, thresholds,
  diagnostics, real-QC, report, map_first, cross-platform, MENDER, workflow summaries). Each is
  rewritten to assert labels are present and the strict column carries the old outcome.

### M2: refused panels are mapped

- A declared panel that is refused today maps nothing (`panel.py:3348`,
  `write_refused_manifest` in `run_annotation.py:1036` and `pipeline.py:3511`). MAP runs anyway
  on whatever genes resolve, and RESOLVE records `qc_panel_trust = refused` with its reasons.
- A refused primary (human `consensus.py:1235,1245`, mode `primary_missing`; mouse
  `:1925-1926`) keeps its calls. A refused SEA-AD second vote stays out of the vote (its calls
  are not trusted for agreement) but its `mmc_*` columns are written.
- The one case with no labels: a panel with no resolved gene. RESOLVE then writes the
  new status `unmappable` and a run-summary line, and the run does not fail.

### M3: downstream steps stop skipping

- Clustering finalize (`clustering/map_first.py:467-530,739-788`): `LeafGate` no longer forces
  leaves to `unresolved`; `ct_branch` / `ct_leaf` / `ct_mender_state` come from the cells' own
  confident labels, and the P3 columns are copied to `obs`.
- MENDER: runs on every dataset that has assigned states (it will, now); its outputs record the
  dataset's `qc_dataset_level`.
- Cross-platform statistics (`clustering/cross_platform.py`): computed at every level, each
  statistic tagged with the levels the two datasets' QC supports (`restricted_by`), instead of
  `none` / `broad_only` caps.
- Cortical depth: unchanged code; the new columns pass through.
- Report: banners stay (they are information); item 4 dotplots and the per-branch dotplots are
  drawn for every dataset, captioned with the advisory level; item 2 leaf shares and item 9 (H12)
  are computed and marked "advisory: dataset failed / broad_only". Acceptance scoring
  (`report_scoring.py:299,578`) reads the strict columns.
- Nextflow end-of-run summary (`AnnotationRunRecord.groovy`, `AnnotationSettings.completionSummary`):
  "broad-only or failed dataset gates" becomes "datasets with QC warnings (labels kept)".

### M4: per-cell counterparts of the dataset signals

- QC stage (`merxen.qc`): per-cell transcript enrichment over the local background, written to
  `*_cell_metrics.csv`; RESOLVE joins it and sets `flag_low_enrichment`.
- RESOLVE: per-cell marker-referee group and disagreement from the G2 computation it already
  does.
- This changes QC outputs, so a resumed run re-runs QC and everything after it once.

### M5: docs, filtering helper, release

- Rewrite the gate contract in `docs/annotation-overview.md` (trust table, "The dataset gate",
  status values, mouse gate), `docs/stages/annotation.md` (trust states, G1 rows),
  `docs/configuration.md`, `docs/outputs.md` (label-table schema), `docs/stages/mender.md`,
  `docs/stages/cortical-depth.md`, and add a "Filtering cells by confidence" page.
- Mark the superseded sections of `robust-celltype-annotation-plan.md` (§3.1, §4.1-4.3, §5.4,
  §7.5, §7.6, §8.8) as replaced by this plan.
- Update `scripts/acceptance/` to the strict columns.
- `merxen.annotation.filtering.select_cells` with tests.
- Release: the meaning of documented columns changes (`ct_<L>_name` is filled where it was
  blank, `exclude_hard` no longer follows the gate, `not_attempted_gate` is gone), so this is a
  **major** release (v2.0.0) with an Upgrading section that maps old readers to the `*_strict`
  columns.

## Delivery

- One short-lived branch from `main` (serious, wide change) and one PR for M0-M3 and M5 (O6);
  CI and reviews as usual. M4 follows in its own change once M1-M3 have been used on real data
  (O5).
- After M1-M3 the sparse mouse section is re-run with `merscope_z_range` `1-7` (its
  build and segmentation re-run because the projection changes) to check the outcome: labels on
  all four segmentations, the advisory level and warnings as today's gate would have set them.
- `-resume` effect: RESOLVE and everything after it re-run (annotation code and config change);
  MAP stays cached except for panels that were refused; M4 re-runs QC.

## Decisions

The owner approved the recommendations on 2026-11-25, except O3 (keep `hard_min_counts` at 10)
and O6 (one PR).

| # | Question | Recommendation | Decision |
|---|---|---|---|
| O1 | Keep the `*_strict` columns, or drop the old semantics entirely? | Keep: one code path, the acceptance scripts and anyone reproducing earlier results keep a column with the old meaning. | Keep. |
| O2 | Should `ct_final_level` / `ct_branch` / `ct_leaf` / `ct_mender_state` (what MENDER and the hierarchy use) follow the cell's own confidence (lenient) or the strict view? | Lenient, as the owner wants labels everywhere; the strict variants are one filter away. | Lenient. |
| O3 | Cells below `hard_min_counts` (10) are not mapped at all. Lower it so more cells get a label (with `flag_low_counts`), e.g. to 3? | Lower to 3 and let the user filter on `total_counts`; 0-count cells cannot be labelled. | Keep 10. |
| O4 | Should a refused SEA-AD second vote still take part in the human vote? | No: write its calls, keep it out of the vote, flag it. | Out of the vote, calls written. |
| O5 | Build M4 (per-cell enrichment and referee) now, or after M1-M3 have been used on real data? | After M1-M3: M1-M3 deliver the owner's requirement; M4 adds new science and a QC re-run. | After M1-M3. |
| O6 | One PR or one per milestone? | One per milestone: each is reviewable and M1 alone already fixes the sparse-section case. | One PR. |
