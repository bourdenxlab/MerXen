# Reference-based annotation (in development)

Reference-based cell-type annotation replaces the legacy marker scoring of
[Squidpy clustering](clustering-squidpy.md) when a species runs in
`map_first` mode. It is being built milestone by milestone
(`docs/plans/robust-celltype-annotation-plan.md`); legacy runs are
unchanged. This page covers what exists so far: declared panels
(`merxen annotation-panel`), the reference bundles that
`merxen annotation-reference-prep` builds into the reference store, the
two pipeline processes that run them (`--annotation_prepare_only`), and the
MAP step (`merxen annotate` and its pipeline process
`CLUSTERING_SQUIDPY_ANNOTATE_MAP`), which maps samples onto the bundles.

## Reference bundles

A bundle holds everything mapping needs from one reference on one declared
panel. Bundles live at `<store>/<reference_id>/<build_hash>/` and are never
modified or deleted automatically (see [CLI reference](../cli.md#merxen-annotation-reference-prep)).

| Reference | Role | What the builder does |
|---|---|---|
| `whb_frontal_supc_clus` | human primary (also the set-c sensitivity bundle, the same builder on the set-c panel) | Copies and verifies the frontal WHB region precompute (ROIs Human A44-A45, A46, A32, ACC; 653 subclusters, 125,481 cells, 59,357 genes), or rebuilds it from the WHB h5ads; truncates it to supercluster → cluster; finds panel reference and query markers (`n_per_utility` 30). |
| `seaad_mr_panel` | human secondary | Copies the SEA-AD Multiregion precompute (release 20260711, 207 supertypes × 36,601 genes; md5 `9b1d5f50a412cd69311fa1175d4836a7`, sha256 `abbd4984bc45…3f6a`) and its small taxonomy tables; panel markers. Missing pinned files are fetched into `<store>/.downloads` (or seeded from a verified local copy). |
| `wmb_panel` | mouse primary | Samples at most `annotation_wmb_max_cells_per_cluster` (50) cells per cluster from the local WMB-10Xv3 h5ads (seed 1; the self-map test cells excluded, see below), restricted to the marker gene universe (see below); builds the marker precompute; finds panel markers with the supertype level dropped; copies the Allen `precomputed_stats_ABC_revision_230821.h5` (md5 `d13b316a1755c459d75b8e45ff92ddfc`) as the mapping precompute. |
| `wmb_region_share` | mouse, panel-independent | Class and subclass × CCF-division shares of the ABC MERFISH-C57BL6J-638850-CCF cells (OB = MOB, AOB, olfactory nerve layer and OLF-unassigned anterior of AP 2.5 mm, split from OLF), each node's home division, per-section compositions and AP composition windows (each section with its neighbours) for every section. |
| `whb_whole_ctx_panel` | human, optional | Whole-WHB panel markers with the 16 cortex-implausible superclusters dropped and the lookup filtered to the pruned tree; refused above 1,000 genes. |
| `whb_frontal_supc_clus_ho` | human, resolvability (M3b) | The frontal WHB cells restricted to the panel, without the held-out donor (`annotation_calibration_holdout_donor`, `auto` = the donor with the fewest frontal cells, H19.30.002); clusters with ≥ 5 training cells; a supercluster → cluster → subcluster precompute of the training cells truncated to supercluster → cluster, with panel markers. The held-out donor's cells (≤ 1,000 per supercluster, all cells of rare ones, ≤ `n_test_cells`) are the test set (`test_cells.h5ad`, native panel counts and truth). E2's HO recipe. Every non-neuronal supercluster is topped up to the same cap with WHB non-neuronal nuclei of E2's 14 neocortical dissections outside the frontal ROIs (user decision 2026-09-27, see the self-map below); `test_source` marks each test cell `holdout_donor` or `other_region`. |
| `wmb_selfmap_testset` | mouse, resolvability (M3b) | The 11,913 self-map test cells plus up to 30 cells per supertype of the non-neuronal classes (Astro-Epen, OPC-Oligo, Vascular, Immune) that the `wmb_panel` marker build did not use (its sample is recomputed with the same rule), restricted to the panel (`test_cells.h5ad`). |

Every bundle records in `bundle.json` the full sha256 of each source, the
copied files, the `cell_type_mapper` steps (input JSON, logs, wall time and
peak RSS under `logs/`) and these diagnostics: levels and node counts,
markers per parent (min / median), weak parents, parents auto-collapsed for
lack of markers and the leaves they hide, lookup keys dropped because the
mapping tree lacks the node, the panel coverage (panel genes in and absent
from the reference, root markers, root children separated) and
`marker_unsupported_nodes`: the mapping-tree nodes no marker of their own
supports (the nodes below an auto-collapsed parent and, for `wmb_panel`, the
nodes without any marker-training cell). `cell_type_mapper` can still assign
these nodes with their ancestors' markers, so RESOLVE (M3) reports such a
label as unresolved at that level; the nodes stay in the mapping tree, as in
the validated runs.

### What `build_hash` covers

`build_hash` is the sha256 of everything a bundle's content depends on, and
nothing else: `ANNOTATION_BUILDER_VERSION` (a test pins the builder code to
it), the builder and its content parameters (including the curated assets'
sha256, `weak_parent_markers` and the 1,000-gene large-panel limit), the
reference id, species, role and taxonomy, each source file's identity (path,
size, mtime, sha256 of the first and last 64 MB; a directory source lists
only the files its builder reads, so the `.lock` and `.tmp` files the legacy
downloader leaves in the shared ABC cache never count), the `panel_hash`,
hierarchy, nodes to drop, drop level, `n_per_utility`, the large-panel
prefilter, the depth grid and the `cell_type_mapper` version and commit.

With resolvability on (builder version 3), the builder parameters of the
primary and secondary bundles also hold what the self-map output depends on:
the recipe and its version, sigma and spill, the test set (its builder,
held-out donor, `n_test_cells`, caps and seeds), the self-map bootstrap
settings and the resolvability code version; the test set's source files are
among the bundle's own sources.

Not in it, and recorded in `bundle.json` instead: the panel's symbols
(bundles hold Ensembl IDs only and match state genes by ID, so a gene list
and a prepared pair panel with the same IDs share one bundle, whatever
symbols they declare; `built_from_panel` names the panel that first built
it), the MAP bootstrap settings, and the RESOLVE-time resolvability
settings (targets and margins, the Wilson, confident-call and coverage
minimums, trust; `recorded_settings`): RESOLVE re-derives the decisions from
the cached self-map cells, so threshold, trust and flag changes re-run only
RESOLVE.

Finished bundles are read-only (files 0444, directories 0555); reuse checks
every file's size and the sha256 of every file up to 64 MB, so an in-place
edit is refused rather than reused.

### Set c

Same-panel human pairs get set c, a cross-platform sensitivity panel
(plan §3.2, D-A7):

- **Seeded set-a family** (the 296-gene E5 panel and its post-M0e 297-gene
  form): set c is set a minus the curated E5 list of 32 platform-deviant
  genes (`setc_exclusions_human.csv`; |median-centred log2 Xenium/MERSCOPE|
  > 2 within confident oligodendrocytes in ≥ 3 of the 4 pairs), the set E5
  validated. It is one family-level panel, the same for every pair (264
  genes on the 296-gene set a, 265 on the 297-gene one), so one set-c bundle
  serves all four pairs. `panel_report.json` also shows what the label-free
  rule would drop on the pair, as a cross-check.
- **Any other family**: the label-free rule, flagged `label_free_rule` in
  `panel_report.json`: drop the set-a genes whose pseudobulk
  log2(mean X / mean M) deviates by more than 2 from the pair median, over
  table cells inside the pair's shared tissue mask. On the E5 pairs this rule
  drops 44-73 genes, including canonical markers (P2RY12, GAD2, VIP), so it
  is a fallback until the family has a curated list. In pipeline runs it
  needs the shared tissue mask from ALIGN (wired in M5) and is refused
  without it; it never uses a published mask looked up while ALIGN may
  still be running.

### Mouse marker gene universe and self-map test cells

`cell_type_mapper`'s raw-CPM normalisation runs over the genes of the
marker-training h5ads, so the marker universe changes the markers. The
validated mouse lookups (E3, E7, MO1) were built on the 899-gene union of
the ag7 and VZG2 panels (`wmb_marker_universe_ag7_vzg2.csv`): a panel whose
genes all lie inside it (ag7, VZG2 and their subsets) uses that union and
reproduces the validated lookups exactly; any other panel uses its own genes
and is recorded as not the validated configuration
(`validated_configuration` in `bundle.json`), to be checked by M3's shadow
run. `annotation_wmb_marker_gene_universe_path` overrides the universe.

The 11,913 WMB self-map test cells (`research/selfmap/truth.csv`, the M3b
resolvability test set) stay out of the marker-training cells. While
`annotation_resolvability` is true a `wmb_panel` build without them is
refused (preflight and builder). A copy whose sha256 matches the pinned file
is seeded into `<store>/.downloads/local/wmb_selfmap/` and the bundle reads
and hashes that copy, so moving the evidence archive never changes
`build_hash`. `bundle.json` records `selfmap_test_cells_excluded`.

### Resolvability self-map (M3b)

With `annotation_resolvability` true (the default) the WHB, SEA-AD and WMB
builders run the self-map of plan §8.3 on their panel
(`merxen.annotation.resolvability`), after their own content:

1. **Test cells.** WHB and SEA-AD get `whb_frontal_supc_clus_ho` from the
   store (built once per panel and shared), WMB `wmb_selfmap_testset`. WHB
   maps the held-out cells onto the held-out bundle, never onto itself
   (its precompute contains the test donor); SEA-AD and WMB map onto the
   bundle being built. *Other-region non-neuronal cells (user decision
   2026-09-27):* the held-out donor holds few cells of the thin
   non-neuronal superclusters (set a: Vascular 23, Fibroblast 5, COP 20),
   so every non-neuronal supercluster (vocab broad class other than
   Neurons and Mixed/Unknown) is topped up to the per-supercluster cap
   with WHB-10Xv3 non-neuronal nuclei of E2's 14 neocortical dissections
   outside the frontal ROIs (`reference.HO_OTHER_REGION_ROI_LABELS`: MTG,
   STG, M1C, A43, A40, V1C, V2, A19, S1C, A1C, A5-A7, ITG, A38, A13; E2
   `research/insilico/01b_extract_nonneurons.py`), any donor, but only of
   clusters the held-out training reference keeps (≥ 5 training cells;
   user decision 2026-09-27, `HO_OTHER_REGION_VERSION` 2 since
   2026-09-28: version 1 drew them from any cluster as E2 did, and 694 of
   the 2,955 set a cells were of clusters the reference lacks, whose
   cluster truth no call can name). The WHB cell metadata
   (`annotation_whb_metadata_dir`) is a test-set source. Candidates among
   the frontal reference cells (training, marker and held-out donor cells)
   are dropped and the draw is checked disjoint from them; `bundle.json`
   (`test_set.other_region`) records the dissections, seed, cap,
   candidates and drawn cells per supercluster, broad class, dissection and
   donor, the cluster rule (`cluster_rule`), the candidates it left out
   (`n_excluded_cluster_not_in_training`, per supercluster and number of
   clusters) and a check that no drawn cell is of an absent cluster
   (`n_cluster_not_in_training` 0). *Truths no call can name (M3b review 2,
   as E2 `research/insilico/02_make_queries.py`):* held-out donor cells
   whose truth supercluster is a sink, has no floor class (Mixed/Unknown:
   Miscellaneous 227 and Splatter 22 cells of H19.30.002, which have no
   broad or supercluster truth, so every call of them was scored wrong) or
   is implausible in the region (Amygdala excitatory, 6 cells: calls to it
   are excluded) are left out before the draw
   (`reference.ho_truth_exclusions`; `test_set.truth_exclusion` in
   `bundle.json`).
2. **Simulation.** Every test cell whose native panel counts reach a grid
   depth `D` is thinned binomially to `D` (human panels up to 1,000 genes
   `[10, 15, 30, 60, 120, 250]`; mouse and larger panels
   `[10, 20, 50, 100, 250, 500, 1000, 2000]`); cells never go up. Recipe
   `R1_contam_HO` (E2): per-gene efficiency LogNormal(0, 0.8) (one draw per
   gene, shared by every cell and depth, median 1) inside the thinning,
   plus 25% of `D` thinned from a random cell of another broad class; truth
   stays the host's. A clean (thinning-only) run is the upper bound. Seed
   0. *Keyed draws (resolvability version 5, 2026-09-28):* every per-cell
   draw is keyed by what it simulates, not taken from one stream over all
   test cells: a simulated cell's thinning and its spill's thinning come
   from a generator seeded by a stable hash (BLAKE2b) of the seed, the
   recipe name and version, the cell id and `D`
   (`resolvability.cell_draw_key`), and the spill partner is the eligible
   cell with the highest rendezvous score of the host's and the
   candidate's keys (`rendezvous_choice`, uniform over the eligible cells).
   The gene efficiency is the pre-registered draw of versions 1-4: one
   generator seeded by `[seed, 0x6566]` over the panel's gene order
   (`resolvability.gene_efficiency`), which depends on the seed and the
   panel only, never on the test cells. So adding, removing or reordering
   test cells leaves every other simulated cell unchanged, except a host
   whose partner left or was outranked by an added cell (its spill only).
   Up to version 4 one stream drew every cell, and the review-2 rebuild (a
   few test cells fewer) redrew all of them, moving 0-7 of 85 validated
   broad / supercluster bins per human dataset. *Version 6 (M3b review 3,
   2026-09-28):* version 5 had also keyed each gene's efficiency by the
   gene id. Neither requested change needed that, and it replaced the
   pre-registered seed-0 efficiency realisation with an unrelated one
   (correlation of the log efficiencies -0.06 on set a); version 6
   restores the pre-registered draw and keeps the per-cell keys.
3. **Mapping** with the production configuration (bootstrap factor 0.5, 100
   iterations, seed 0, raw normalisation, the bundle's lookup).
4. **Levels.** WHB: lineage, broad, NT (group probabilities summed over
   same-group runner-ups), supercluster and the report-only cluster; SEA-AD:
   its 7-class call (E2 definition) against the WHB truth; WMB: broad, class,
   NT, subclass and the report-only supertype (cluster calls aggregated to
   their supertype). Calls are keyed by the called node's class: the E2
   floor classes for human (neurons by NT, COP apart at supercluster level),
   WMB class names for mouse. Calls to sinks and region-implausible WHB
   superclusters are excluded (never confident in production), and so are
   WHB broad calls that production keeps at lineage by the §5.2 COP rule
   (supercluster call COP below the 120-count COP floor or below
   supercluster bp 0.69; `whb_cells_rules`, applied to the cells table
   before the decisions; SEA-AD's confident-OPC rescue is not simulated,
   so broad OPC is measured conservatively).
5. **Local threshold rule** (E2 verdict 3; never the set-level rule): per
   (level, class, depth bin) an isotonic map of correctness on bp is fitted
   on one split half of the test cells (`crc32(cell id) mod 2`); the
   threshold `t*` is the lowest bp ≥ the v1 raw threshold at which it
   reaches the target (capped at 0.99), so thresholds only ever rise. The
   other half checks it: a bin is emitted with ≥ 50 confident calls, a
   Wilson 95% lower bound of their precision (on the Kish n) ≥ target −
   0.02, a point precision ≥ target (added 2026-09-27, one rule with the
   gate-P evaluation rules) and coverage ≥ 0.2. The `validated` regime
   applies the pre-registered default, which is fitted on no test cell, so
   it checks the same rule on every call of the bin (both halves); its
   fit-half `t*` is only reported.
6. **Pooled deep bins and extrapolation** (user decision 2026-09-27,
   replacing the `D_max` inheritance). Only cells whose native counts reach
   a depth are thinned to it, so deep bins run short of calls. A bin with
   fewer than 50 confident calls on its check set takes the verdict of the
   deep-end pool: from the deepest grid bin, bins are pooled into a "≥ d"
   set, each test cell counted once at its deepest bin (taken over all its
   calls at the level, before any class or confidence filter, so a
   confident shallow row never stands in for an unconfident deep one),
   until the set holds 50 confident calls (or a fitted regime's isotonic
   fit shows no threshold reaches the target); its shallowest bin is
   `D_P`. The set is tested with the same rule (its own fit and `t*`, the
   target of `D_P`, the Wilson bound on the Kish n). Every bin at or above
   `D_P` that holds 50 confident calls itself keeps its own verdict and is
   withdrawn when the pool fails (reason `pool_<reason>`): a pooled pass
   never overrides a bin's own failure (M3b review 2; before, only `D_P`
   did). The bins that are not judged on their own take the pool's verdict
   and are marked `pooled`; those deeper than `D_P` are also marked
   `resolvability_extrapolated`. Only calls with a positive weight count
   towards the 50 (RESOLVE: a bin of calls of types the dataset lacks is
   not judged on its own). A
   class that never reaches 50 confident calls even with every bin pooled
   is `not_resolvable` (`insufficient_calls`) wherever it is not judged on
   its own; a class without `D_max(c)` (the deepest grid depth with ≥ 50
   test cells of class `c`) is never emitted (`too_few_test_cells`; no
   pooling across classes). `gate_p_tested_sets` uses the same pooling
   with `n_min` 200. Everything not emitted is `not_resolvable`; the
   report-only fine levels are emitted only with
   `annotation_allow_fine_levels` and a seed-1 re-map that changes ≤ 2% of
   their confident labels (`level_emission`).
7. **Regimes.** `validated` (real-data-validated families: base targets,
   the pre-registered default applied, local thresholds that would raise
   it listed for H18), `provisional` (other panels and simulation-validated
   families: targets + 0.05, + 0.10 below 60 counts, capped at 0.97; `t*`
   applied) and `trust` (base targets with the local rule).
8. **Floors** are the smallest emitted depth; for panels without real-data
   validation the applied floor is max(known floor, simulated floor; known
   = the maximum over platforms and panels), and at least 60 for the mouse
   subclass level. Validated panels keep the packaged floors per platform:
   the summary lists them per (platform, panel family) with
   `floor_source` `packaged` and no single floor.
9. **Trust constraint:** `refused` when the broad level is emitted for no
   class at any depth ≤ 250 (`trust` regime), `broad_only` when the leaf
   level is emitted for fewer than half of the classes with `D_max` at
   every such depth. The constraint becomes the bundle's `panel_trust`;
   otherwise the panel diagnostics and `validated_panels.csv` decide
   between `provisional` and `validated`.

PREP's decisions are unweighted and set only the trust constraint. RESOLVE
(M4) re-derives them with the simulated cells reweighted to each dataset's
soft composition (`ResolvabilityTables.decisions(composition=...)`,
`composition_weights`; plan §8.3 "within c"), with this scheme (M3b
review):

- **Per depth bin.** The simulated cells of a depth bin follow the
  composition of the dataset's table cells in that bin
  (`DatasetComposition.from_cells`: each cell's assigned bp and runner-up
  probabilities, summed per type and bin); a bin with fewer than
  `composition_min_bin_cells` (50) cells uses the dataset's overall
  composition. Low-count cells are often called into a few types (COP),
  so one dataset-wide composition over-weighted those types at every
  depth.
- **Rare types at broad-class level.** A truth type with fewer than
  `weight_min_type_cells` (20) test cells in the bin takes the weight of
  its broad class's common types (`sum q / sum p` over them; the class's
  pooled weight when it has no common type), so a handful of test cells
  cannot stand for a large composition share.
- **Trimmed per tested set** (resolvability version 4, M3b review 2).
  Within each tested set (a (level, class, depth) bin's calls, or a
  pooled deep set) weights above `weight_trim_factor` (10) times the set's
  median positive weight are capped before the fit and the checks
  (`RuleSettings.weight_trim_scope = "judged_set"`). Versions 2-3 capped
  at 10 times the median of the whole depth bin, which its commonest test
  types (neurons) set: in deep bins every glial type hit the same cap and
  reweighting did nothing inside a called glial class (P7513 MERSCOPE
  broad Oligo at 120: COP and Oligodendrocyte both weighed 6.45). Bundles
  of those versions keep the per-bin trim when re-derived.
- **Recorded.** Every decision carries the Kish effective n (the n of the
  Wilson bound) and the largest single call's share of the confident
  weight (`max_weight_share`).

On the eight human shadow datasets, against the stage C weighting (one
dataset-wide composition, no pooling or trim) on the same rows, this lifts
the smallest Kish n of the reweighted broad bins with ≥ 50 confident calls
from 3.9–20.5 to 85–171, and the largest single-cell weight share falls
from up to 36% to at most 3.5% (broad) and 5.1% (supercluster); on ag7 no
emitted bin has a cell above 16% of its weight
(`m3b/review_fix/REVIEW_FIX_REPORT.txt` in the evidence archive). Types
absent from a dataset get weight 0, so a class the dataset (almost) lacks
is not emitted there.

**Measured on the validated panels** (M3b, 8 processes on the shared host;
`m3b/resolvability/` in the evidence archive):

| Bundle | Test cells | PREP wall (peak RSS) | Resolvability part | Trust |
|---|---|---|---|---|
| set a WHB (297 genes) | 9,032 of the held-out donor's 32,606 (≤ 1,000 per supercluster; 2,425 reach 250 counts) | 14.1 min (2.8 GB) | held-out bundle 10.4 min (3.6 min extraction, 6 min source checksums) + self-map 1.9 min | `validated` |
| set a SEA-AD | the same (held-out bundle reused) | 6.4 min (3.0 GB) | self-map 2.7 min | `validated` |
| ag7 WMB (500) | 13,059 (11,913 + 1,146 non-neuronal; 61% reach 1,000 counts, 17% 2,000) | 63.7 min (20.2 GB, query markers) | test set 1.1 min + self-map 11.8 min | `validated` |
| VZG2 WMB (815) | 13,059 (70% reach 1,000 counts, 27% 2,000) | 65.0 min (27.1 GB, query markers) | test set 0.8 min + self-map 14.5 min | `validated` |

All within the §8.3 budget (human ~35, mouse ~25 min) and the PREP limits
(≤ 2.5 h human, ≤ 2 h and ≤ 40 GB mouse); none of the four bundles gets a
resolvability constraint. Resolvability version 2 (M3b review) gives the
self-map bundles new build hashes: the set a WHB and SEA-AD bundles were
rebuilt in 1.6 and 2.0 min (the held-out test-set bundle is reused), the
set c WHB bundle took 3.9 min with its own held-out test set; the rebuilt
set a tables equal the v2 decisions re-derived from the stage C cells
(`m3b/review_fix/VERIFY_REBUILD.txt`). The ag7 and VZG2 cells tables do not
change with version 2, so those bundles were not rebuilt; a pipeline PREP
rebuilds them (new `build_hash`, about an hour each). Re-deriving the
decisions reweighted to a dataset's composition takes seconds. Applied to
the M3 shadow MAP runs of the eight human datasets (reweighted per depth
bin, validated regime, version 2), broad is emitted for 71–85% of table
cells and supercluster for 69–83% (0–13% of table cells in bins inherited
from `D_max`; stage C, version 1: 59–82% and 59–83%); ag7 proseg_hybrid:
class 97%, subclass 98% (version 1: 86%, 85%). The H18 exceptions are
listed under Known limitations.

**Resolvability version 3** (the user's H18 decisions of 2026-09-27: pooled
deep bins with the point precision in the rule, other-region non-neuronal
human test cells; `m3b/followup/H18_FOLLOWUP.txt` in the evidence
archive) gives every self-map bundle a new build hash. Rebuilt at 8
processes: set a WHB 4.3 min (its new held-out test set 2.6 min, of which
the other-region draw 7 s), SEA-AD 2.1 min, set c WHB 4.1 min, ag7 37.1 min
(20.2 GB) and VZG2 45.1 min (27.1 GB). The set a test set holds 11,987
cells: the donor's 9,032 plus 2,955 other-region nuclei (Microglia 757,
Vascular 655, OPC 503, Fibroblast 433, COP 264, Astro 210, Oligo 133; none
among the 125,481 frontal cells; 694 of clusters the held-out training
reference lacks), so Vascular, Fibroblast and COP have 678, 438 and 284
test cells (were 23, 5, 20). The mouse cells tables are unchanged (same
test set and markers). Set a (PREP) now emits broad Vascular at 10–250 and
Fibroblast at 15–250 counts. Reweighted to the eight human datasets, broad
is resolvable for 67–97% of table cells and supercluster for 81–95%, 0–7%
of table cells in pooled (extrapolated) bins; ag7 proseg_hybrid class 97%,
subclass 98% (2% pooled). RESOLVE reweights each pooled set as one set to
the dataset's cells at `>= D_P` (`pooled_composition_weights`): with the
per-bin weights, a deep bin where a dataset lacks a type let one row carry
a whole set (ag7: 69 calls, Kish n 1). Reweighted decisions take 1.5–1.8 s
per human dataset and 3.2–3.6 s on ag7.

**Resolvability version 4** (M3b review 2, 2026-09-28;
`m3b/review2/REVIEW2_H18.txt`) gives every self-map bundle a new build
hash again. Rebuilt at 8 processes: set a WHB 4.2 min (held-out test set
included), SEA-AD 2.1, set c WHB 4.1, ag7 36.7 min (20.2 GB) and VZG2
44.7 min (27.1 GB). The set a test set holds 11,732 cells (the donor's
8,777 without Miscellaneous 227, Splatter 22 and Amygdala excitatory 6,
plus the same 2,955 other-region nuclei); the mouse cells tables are
identical to version 3. Every stored summary equals the decisions RESOLVE
re-derives from the stored cells (0 differing bins in the five bundles).

**Resolvability version 5** (M3b final follow-up, 2026-09-28;
`m3b/final_followup/FINAL_FOLLOWUP.txt`; superseded by version 6): the
keyed per-cell draws, the gene efficiency keyed per gene id (undone in
version 6) and the training-cluster rule for the other-region cells give
every self-map bundle a new build hash. Rebuilt at 8 processes: set a WHB 4.3 min
(held-out test set 2.5 min), SEA-AD 2.2, set c WHB 4.2, ag7 36.9 min
(20.2 GB) and VZG2 45.1 min (27.1 GB). The keyed simulation takes 7-9 s
per human and 11-12 s per mouse recipe (one generator per simulated cell;
version 4: 0.2-2.5 s). The set a test set holds 11,040 cells: the donor's
8,777 plus 2,263 other-region nuclei; the rule left out 852 candidates of
27 clusters the held-out training reference lacks, so Vascular,
Fibroblast and COP have 412, 144 and 152 test cells (version 4: 678, 438,
284) and Fibroblast's `D_max` is 60. Every stored summary equals the
re-derived decisions. Simulating the version-4 and version-5 test sets
with the same code, the 44,427 simulated cells of their 9,654 shared test
cells are identical with the clean recipe; with `R1_contam_HO` 19,545
differ, every one with a new spill partner (19,630 changed partner; 85 of
them give identical counts), and the host thinning is identical for all.
Partners change that often because the other-region top-up
redrew 1,386 of its cells when its candidate pool changed: the test-cell
selection is still one draw per supercluster. MapMyCells also draws its
bootstrap per chunk of query cells, so a changed query moves the
bootstrap probabilities of other cells: mapping the same simulated cells
in another order changed 67% of the bp values, 5% of the broad
confident-at-0.73 flags and 0.02% of the confident broad labels, and 0-2
of the 83 validated broad / supercluster bins per dataset (a full redraw,
seeds 1 and 2: 0-7).

**Resolvability version 6** (M3b review 3, 2026-09-28;
`m3b/final_followup/review3/H18_V6.txt`, summarised in
`m3b/final_followup/FINAL_FOLLOWUP.txt` §12): the gene efficiency is the
pre-registered version-4 draw again, with the per-cell keys and the
training-cluster rule of version 5, so version 6 is the outcome of the two
requested changes alone. Rebuilt at 8 processes on a loaded host: set a
WHB 2.9 min (the version-5 held-out test set reused), SEA-AD 4.9, set c
WHB 2.4, ag7 43.5 min (20.2 GB) and VZG2 69.4 min (27.1 GB). Every stored
efficiency vector equals the pre-registered formula and the version-4
bundle's; every stored summary equals the re-derived decisions; the set a
decision rows equal the final follow-up's offline "e4" simulation (version-5
per-cell keys with the version-4 efficiency) and an offline re-map of the
stored simulation, call for call. Simulating the version-4 and version-6
test sets with version 6, the clean recipe leaves the 44,427 shared
simulated cells identical; with `R1_contam_HO` 19,552 differ, every one
with a new spill partner (19,630 changed partner; 78 of them, all at 10-15
counts, give identical counts, 60 with an empty spill from both partners),
and the host thinning is identical for all.

### Bundle files

| File | Content |
|---|---|
| `mapping_precompute.h5` | Precompute MapMyCells maps onto (a checksummed copy or a derived file). |
| `source_precompute.h5` | WHB: the verified copy of the region precompute (subcluster leaves). |
| `marker_precompute.h5`, `marker_training_cells.csv` | WMB: the panel-restricted marker precompute and the cells it was built from. |
| `query_markers.json` | The lookup as `cell_type_mapper` wrote it. |
| `query_markers.filtered.json` | The lookup restricted to the mapping tree and the panel; auto-collapsed parents keep an empty entry. |
| `mapping_tree.json` | The mapping tree (after `drop_level` and `nodes_to_drop`). |
| `profiles.parquet` | Per node × panel gene (Ensembl ID): detection fraction, mean log2(CPM + 1), expected CPM (conditional log-normal from `sum` / `sumsq` / `gt0`) and its share of the panel. |
| `negative_genes.parquet` | Primary references: per broad class × panel gene (Ensembl ID), the detection fraction in each reference and whether the gene is negative (< 1% in every reference, not a curated state gene; state genes matched by ID). |
| `vocab_snapshot.csv` | Every mapping-tree node with its vocabulary broad class, NT and flags. |
| `depth_grid.json` | The resolvability depth grid of the panel. |
| `resolvability.parquet` | Primary and secondary bundles (resolvability on): one long table, `kind` per row type: `bin` (per recipe × level × class × depth: calls, precision and coverage at the default threshold, the local threshold), `curve` (precision and coverage at thresholds 0.50–0.99), `isotonic` (fit-half knots), `node` (per-node precision, recall, F1), `confusion` (truth × call within the called class), `decision` (the three regimes below), `gene_efficiency`. |
| `resolvability_cells.parquet` | One row per simulated cell × level × recipe: parent class, depth, split half, call, bp, `avg_correlation`, truth, truth class, truth leaf, correct. RESOLVE reweights these to each dataset's composition. |
| `resolvability_summary.json` | Recipes, levels, settings, test-set counts, `D_max` per level and class, emission per regime, level, class and depth (status, threshold, `t*`, extrapolated, pooled, reason), the pooled deep sets per regime, level and class (`D_P`, the bins taking their verdict, statistics, `t*`, status), floors, validated thresholds the local rule would raise (own bins whose fit half holds 50 calls; pooled sets carry `would_raise`), the validated bins whose fit half holds fewer (`validated_thresholds_not_evaluable`: `t*` unknown, never counted as a raise), the trust constraint, fine-level seed stability, runtimes and mapping runs. |
| `test_cells.h5ad`, `test_cells.parquet` | Resolvability test-set bundles: native panel counts of the test cells and their truth per level (`truth__<level>`), composition key and spill group; human: donor, dissection and `test_source`. |

### Declared panels, gene IDs and controls (M3b)

`merxen annotation-panel` (and MAP) read each platform's **declared** panel:
a vendor panel file (Xenium `gene_panel.json`, a MERSCOPE codebook, a gene
table) or the unfiltered `var` of the prepared H5AD. A published clustered
H5AD declares the features its control filter recorded
(`uns["merxen_clustering_squidpy"]["control_feature_filter"]`: kept and
removed), not the `min_cells`-filtered `var`, so zero-count probes and
`min_cells` filtering never change `panel_hash` (measured: the P7513 /
P1212 and ag7 declared hashes equal the M3 shadow ones). The features
`min_cells` dropped are no longer in `var`: on a native-ID platform their
IDs come from the platform's panel file (`merxen annotate
--declared-ids-file XENIUM=gene_panel.json`; a symbol can give another ID,
e.g. GGT1 or H2AFX on the 5K panel), else they are resolved by symbol,
logged and listed as `declared_ids_incomplete` in `panel_report.json`. MAP
maps them as query genes with zero counts (`undetected_declared_genes` in
the run record), as the prepared H5AD would give them almost everywhere,
so they never count as missing panel genes or trigger a subset bundle (M3b
review 2; P5011 reseg MERSCOPE: 32 of 300 declared genes dropped, which
had requested an own-family subset bundle).

A MERSCOPE codebook is read as Vizgen writes it: data rows may end with a
trailing comma (the ag7 codebook VA00282), and its `id` column holds
transcript identifiers (Ensembl transcripts, RefSeq accessions, a UUID for
a custom transgene) that are recorded (`recorded_id`) but never native gene
IDs and never counted by the prefix rules below; only an `id` that is an
Ensembl gene ID is used natively.

**Controls** (`annotation.panel.ControlRegistry`, on the shared registry
`merxen.control_features`): a feature type decides alone when the source has
one (`feature_types`, the type a Xenium `codeword_category` or `is_gene`
implies, a `gene_panel.json` descriptor); only `Gene Expression` is a gene.
Without a type, the documented anchored names are controls (MERSCOPE
`Blank-N`; Xenium `NegControlProbe_`, `NegControlCodeword_`,
`UnassignedCodeword_`, `DeprecatedCodeword_`, `Intergenic_Region_`,
`GenomicControl`, `BLANK_`, `antisense_`), then the `CONTROL_TOKENS`
substring rule, which never removes a feature carrying a native Ensembl ID
or resolving to a gene of the reference gene table. On the current panels
the registry removes exactly what legacy `remove_control_features` removes.

**Gene IDs** (`annotation.gene_ids`; first hit wins, the source is
recorded): the native Ensembl ID of the `var` column (`ensembl_id`,
`gene_ids`, `gene_id`, `feature_id`; version suffix stripped; a native ID
missing from the local table whose symbol the table knows takes the
table's ID, `symbol_fallback`), the pair's other platform (same symbol),
the run species' local gene table (`annotation_<species>_gene_table`, else
`annotation_gene_id_fallback_csv`; exact case first, then a unique
case-insensitive match), an optional alias table (single-target aliases
only) and the curated overrides (`gene_id_overrides_<species>.csv`, each
with a reason). The alias table resolves a panel symbol both as an alias
or previous symbol and as a current approved symbol whose gene table still
lists a previous one (H2AX / H2AFX: the HGNC row's own ID, or the table's
ID of its previous symbol); a symbol pointing to two IDs stays unresolved.
Ensembl gene IDs are upper-cased (`ensg00000141510` is a human ID). Features resolving to one ID are merged and summed at MAP;
unresolved ones (reporter genes, isoform probes such as the Xenium MAPT 3R
/ 4R probes, genes a table lacks) are listed in `panel_report.json`.

**Refusals.** The declared panel is refused, and gets no bundle, when:
- the exact-case species test fails (`species_mismatch`): the panel symbols
  match the other species' table in exact case more often than the run
  species' table, or fewer than half of the run species' case-insensitive
  matches are exact (HGNC symbols are upper case, MGI symbols are not). On
  the current panels: ag7 and VZG2 run as human are refused although the
  case-insensitive fallback would map 487 / 500 and 788 / 815 of their
  symbols to human IDs; the P7513 human panel run as mouse is refused;
- fewer than 95% of the non-control features resolve (`gene_id_resolution`);
- fewer than 95% of the native ID values are Ensembl gene IDs of the run
  species (`native_id_prefix`; symbols stored as IDs, the ag7 failure that
  found 0 of 498 root markers, are listed as `symbols_as_ids`), or more
  than 5% carry another species' prefix (`other_species_ids`). Ensembl
  transcript IDs (MERSCOPE codebooks, isoform probes) do not count.

`panel_report.json` records per declared panel the status and reasons, the
species test (exact and case-insensitive matches per species), the source
of every resolved feature, merged duplicates, unmapped features, release
drift and the sha256 of the resolution table. The annotation panels carry
each declared panel's resolution identity (`declared_resolutions`: the
resolution table's sha256 and the gene tables consulted), and `bundle.json`
records it in `built_from_panel` with the declared panel hashes. It is
provenance only: bundles are keyed by the resolved IDs, so two resolutions
that give the same IDs share one bundle (plan §8.4 puts the resolution
sha256 into `build_hash`; M3b records it instead, a deviation).

### Large panels (M3b)

Panels above `large_panel_genes` (1,000; e.g. the Xenium Prime 5K panels)
are built like any other (plan §8.7), with these differences:

- **Large store, kept reference markers.** Bundles go to
  `annotation_reference_store_large` (dwight:
  `/srv/storage/MerXen/annotation_references_large`) and keep the family's
  reference markers (`reference_markers/`), so a dataset missing some panel
  genes needs only a query-marker step (plan §8.7, D-G7). Bundles of panels
  up to 1,000 genes delete their reference markers as soon as the query
  markers exist (sha256 kept in `bundle.json`).
- **Resources.** PREP gets `annotation_prep_large_memory` (64 GB, from the
  5K measurement below) and 24 h, MAP 48 GB (plan §3.3). The whole-WHB
  optional bundle stays refused above 1,000 genes.
- **Wide depth grid** `[10, 20, 50, 100, 250, 500, 1000, 2000]` for either
  species.

**The 5K memory measurement** (M3b, OD-E8; Xenium Prime 5K Mouse, 5,006
genes, all in WMB; `wmb_panel`, 14.06 M taxon pairs, `--n_processors 8`,
`--max_gb 40`, run alone in a memory-capped scope; evidence
`m3b/simulate/xenium_prime_5k_mouse/`):

| Step | Unfiltered (5,006 genes; default) | Prefiltered (1,992 candidates) |
|---|---|---|
| Reference markers | 2,845 s; largest process 5.8 GB, process tree PSS 20.9 GB; h5 14.2 GB | 2,844 s; 4.9 GB, 27.6 GB; h5 12.3 GB |
| Query markers | 1,290 s; largest process 21.3 GB, tree PSS 20.5 GB | 1,241 s; 37.7 GB, 37.4 GB |

Neither time nor memory grows with the genes up to 5,006 (the evidence runs
gave 20.2 / 25.8 GB at 500 / 815 genes), so the plan's ~120–170 GB estimate
does not hold, and PREP for large panels keeps the standard 64 GB (the peak
37.7 GB + 30% = 49 GB). The production (unfiltered) 5K bundle took 92 min
(reference markers 2,731 s, process-tree PSS 27.2 GB; query markers 1,156 s,
21.3 GB; self-map 23 min on 103,297 simulated cells, MapMyCells about 430 s
per recipe for 103k cells × 5,006 genes at 8 processes, peak RSS 3.3 GB) and
holds 16.2 GB, 13.2 GB of it reference markers; its decisions equal those of
the measurement's unfiltered mapping in all 1,360 bins.

**Per-parent marker prefilter** (`merxen.annotation.prefilter`;
`large_panel_marker_prefilter = "per_parent_topk_union"`, cap
`large_panel_prefilter_cap` 2,000), **opt-in** since the measurement: it
saves neither memory nor time at 5K, and version 1 failed its validation
(the same simulated cells mapped with both lookups agree >= 0.994 per class
at broad, class and NT, but 0.920–0.949 at subclass in 13 of 34 classes,
below the required 0.95). When on, marker discovery runs on at most 2,000
candidate genes chosen per parent of the marker tree (after `--drop_level`),
never by a global variance ranking, which would drop the markers of rare
leaves: every pair of siblings scores each gene by its log2 fold scaled by
cell_type_mapper's penetrance terms (detection of the higher sibling against
0.5, detection contrast against 0.7; minimums 0.8 / 0.1 / 0.1) and keeps its
60 best genes as candidates; each parent orders its genes by max-min greedy
pair coverage (the gene serving the most of the pairs with the fewest chosen
candidates first, up to 30 per pair, the `n_per_utility`), and `k` is the
largest per-parent top-k whose union fits the cap (k = 69 on the 5K mouse
panel). The candidate set (`marker_prefilter.json`: `k`, genes, sha256,
per-parent counts) is the panel stub of `reference_markers` and
`query_markers`; profiles and negative genes still use every panel gene;
method, version and settings enter `build_hash` of every builder that finds
markers. Without it, a large `wmb_panel` whose predicted query-marker peak
(the measured 37.7 GB envelope up to 5,006 genes, scaled with the genes
beyond) plus the OD-E8 margin of 30% exceeds the PREP reserve (`--max-gb` /
0.625) is refused before anything is built (with the 64 GB reserve: panels
above about 6,500 genes): the prefilter is mandatory above the reserve
(OD-E8). No pipeline parameter switches the prefilter on: pipeline runs
raise `annotation_prep_large_memory`, and the prefilter is set only through
`--annotation-config` of the standalone `annotation-reference-prep` and
`annotation-panel-simulate`. The store may then hold a bundle with and one
without the prefilter on the same panel; a standalone store lookup
(`merxen annotate --store`, the subset-bundle finder) takes the one whose
`build_hash_payload.large_panel_prefilter` matches the run's config.

### Panel simulation (`annotation-panel-simulate`, M3b)

The design aid of plan §8.8 predicts what a panel resolves before any of its
data exist (a vendor panel, a custom panel before ordering). It resolves the
gene list as a declared panel, builds the species' self-map references
(human `whb_frontal_supc_clus` and `seaad_mr_panel`, mouse `wmb_panel`)
through the store with the production configuration — the bundles a
pipeline PREP would use, reused when they exist — and reads the predicted
emission per (level, class, depth bin) from their resolvability decisions in
the panel's regime (`provisional` with its margins unless the panel is of a
real-data-validated family), with the trust state, weak and collapsed
parents, runtime, peak memory and disk ([CLI](../cli.md#merxen-annotation-panel-simulate)).
For prefiltered panels it also maps the same simulated cells with the
unfiltered lookup and compares the calls per (level, class) (agreement
>= 0.95 at bp >= 0.8 for every class with >= 50 unfiltered confident calls
at an emitted level, and no parent of the prefiltered lookup below 5
markers: the pre-registered rule of plan §8.7 / NP9, whatever the
unfiltered lookup gives; the relaxed reading "no parent made weak by the
prefilter" is reported as `no_parent_made_weak`, not applied). The pinned
public 10x panel lists
come from `merxen annotation-panel-fetch`. `--gate-p` is the M13 hook for
the gate-P programme; it is refused until M13 registers it.

**Measured on the four public 10x panels** (M3b, 8 processes on the shared
host, the production configuration, resolvability version 4;
`m3b/review2/simulate/` in the evidence archive, `m3b/simulate/` for
version 3; classes emitted per depth bin in the `provisional` regime, of 8
broad / 9 supercluster human and 34 mouse classes, with enough test cells
for 8 / 9 / 24 of them):

| Panel | Genes | Primary level: classes emitted per depth | Wall (from scratch) | Peak (largest process) | Disk |
|---|---|---|---|---|---|
| Xenium Human Brain v1 | 266 | broad 6 / 5 / 7 at 10 / 15 / 30+; supercluster 5 / 5 / 7 / 8 at 10 / 15 / 30 / 60+ | 6.0 min (WHB + SEA-AD) | 2.9 GB | 1.9 + 0.4 GB |
| Xenium Prime 5K Human | 5,001 | broad 0 / 3 / 7 / 7 / 8 at 10 / 20 / 50 / 100 / 250–500, 7 above; supercluster 1 / 6 / 7 / 8 at 20 / 50 / 100 / 250–500 | 8.3 min | 2.9 GB | 1.9 + 0.5 GB |
| Xenium Mouse Brain v1 | 248 | class 3 / 12 / 21 at 50 / 100 / 250+; subclass 4 / 19 / 21 | 25 min | 8.3 GB | 1.4 GB |
| Xenium Prime 5K Mouse | 5,006 | class 5 / 16 / 24 at 100 / 250 / 500+; subclass 3 / 18 / 23 | 90 min | 21.3 GB (tree PSS 22.9 GB) | 16.2 GB (13.2 GB reference markers) |

Version 4 left the held-out Miscellaneous / Splatter cells out of the human
test set: their calls to Exc had been scored wrong, so the 5K human WHB
broad Exc was emitted only from 250 counts and SEA-AD broad Exc never (now
from 20 and 50); the mouse panels do not change (same test set). These
four panels' bundles (the 5K ones in the large store) were not rebuilt for
resolvability versions 5 and 6, so they keep version-4 self-map tables
until they are; a standalone MAP on them logs that its tables are stale.

A 5K pan-tissue panel needs more counts than a brain panel for the same
classes (mouse class at 100 counts: 5 of 24 classes vs 12 for the 248-gene
brain panel and 24 for VZG2), because its counts spread over many genes that
do not separate brain types. All four panels are `provisional` (own
families).

**Checked against real 5K data** (phase 1, 2026-09-28; `5k_real/SYNTHESIS.txt`
in the evidence archive). The "at the expected median depth" line puts every
cell at one depth. On the one public Xenium Prime 5K mouse section (63,147
vendor cells, median 1,089 counts; 250 counts is its 6.7th percentile) the
M3b headline at 250 counts under-predicted real coverage by .13 (class) to
.22 (subclass under the bundle's provisional decisions) and neurons by up to
.52, while it over-predicted glia. Read the per-bin table against the
panel's real per-class depth. Plan milestone M3c replaces the headline
with per-class depth profiles (resolvability version 7, plan §8.3).

**Per-class depth and simulation inputs (M3c, simulation side).** The
headline of `annotation-panel-simulate` is now the per-class depth profile
(`--depth-profile` per-class or pooled CSV, or `--depth-profile-asset`):
per (level, class), the profile's share of the class's cells in emitted
bins and the predicted coverage over them, and, in profile mode, cells
simulated at the profile's per-class TOTAL depth for each ensemble member,
mapped and tabulated per called class (member mean and range). The
expected median depth stays as a secondary line. Profile mode reproduces
phase 1's D3 draws exactly (same simulated cells, hosts, partners and
totals drawn; 98.6% of realised totals identical, the rest from the exact
thinning). Public vendor data enter only as simulation inputs with
provenance (`assets/annotation/sim_inputs/`, OD-E1 amended): the Prime 5K
mouse factor table against WMB 10Xv3 (the `R3_measured_HO` member; its
measured factor only for informative genes whose top class holds >= 0.5% of
the section, every other gene a keyed resample of the measured
distribution), the public section's per-class depth profile ("XOA 3.0
vendor segmentation, public 10x section"), the lung 5K / v1 ratios (a
cross-tissue stress recipe for human Prime only) and the lung FFPE depth
scenario. No asset raises a trust state, no depth profile crosses species,
and version 6 (set a, ag7, VZG2, P5011 MERSCOPE) is byte-identical.

**Resolvability version 7 in PREP (M3c, decision side).** The version is
chosen per panel family: 6 for the families of `validated_panels.csv` (set a
with set c, ag7, VZG2; by the family after trust inheritance or the listed
panel hash) and the pins of `resolvability_v6_pins.csv` (P5011 MERSCOPE);
7 for every other family (Xenium Prime 5K mouse and human, custom and unknown
panels, the M13 custom MERSCOPE panel). `annotation.resolvability.version`
(`auto`) may force 6; forcing 7 on a version-6 family is refused. A
version-6 family's `build_hash` and outputs are unchanged; a version-7
family's `build_hash` holds its members, assets (sha256), chemistry, grid and
top-up rule, so its bundles are new build directories. A version-7 self-map:

- tops the test set up before mapping: every class of the leaf's parent level
  (mouse WMB class; human supercluster-level class, COP apart) with fewer
  than 200 test cells gains cells where its pool allows, stratified by leaf
  type, lowest keyed draw first. Mouse pool: 10Xv3 cells neither
  marker-training nor test cells, only of clusters with at least 5
  marker-training cells, at most 5% of a cluster's cells. Human pools: the
  held-out donor's other cells, then (non-neuronal classes) other-region
  cells of training clusters; never another frontal donor. `bundle.json`
  (`test_set.class_top_up`) records each class's cells before and after,
  what each pool held and gave and whether it ran out;
- simulates and maps each member in turn: eight emission members (since the
  amendment of 2026-09-29, an orchestrator decision pending the user's
  confirmation; plan §8.3 v7.3, pre-registration §15) -- `R1_contam_HO@0`,
  `@6`-`@10` and `R3_measured_HO@2`, `@3` where the species x chemistry has a
  measured factor table (Xenium Prime 5K mouse), else `R1_contam_HO@0`,
  `@6`-`@12` (Xenium Prime 5K human, custom, MERSCOPE and unknown panels);
  `clean@0` reported; `R1_xtissue_lung_stress@0` reported for human Prime,
  never an emission member; 13 grid values above 1,000 genes. The bundles
  built in stage D keep their members (`R1_contam_HO@0`-`@2`, plus
  `R3_measured_HO@0` for 5K mouse). `resolvability.ensemble_r1_seeds` /
  `ensemble_r3_seeds` override the members;
- decides each member by the version-6 rule plus the saturated-bp rule (a
  set without a local threshold whose fit-half calls are more than 90% at
  bp = 1 is judged at the 0.99 cap), then the ensemble: a bin is emitted when
  the union of the members' calls passes the §8.3 rule at its own threshold
  with each test cell counted once (E1), and every member emits it or the
  member precisions lie within max(0.03, 3.5 SE) with at least 10 calls each
  and the pooled Wilson bound clears target - 0.02 by one standard error
  `sqrt(p (1 - p) / n_eff)` of the pooled precision (E2; the margin since the
  amendment of 2026-09-29, `ensemble_spread_margin` when it fails; bundles
  built before it re-derive without it); bins short of calls take the
  ensemble's deep pool, whose spread route needs the same margin;
- fills a bin deeper than the shallowest emitted one when its measured point
  precision and coverage pass (only the power conditions are waived), never
  for non-neuronal classes at 1,000 counts or more (those bins are marked
  `nonneuronal_high_depth`); floors and trust come from the decisions before
  the fill. When an emission member uses a simulation-input asset
  (`R3_measured_HO`), the trust constraint is the more severe of the full
  ensemble's and that of the asset-free members (the R1 draws, decided on
  the same cells), so an asset can lower the trust state but never raise it
  (`trust_asset_guard` in the summary; pre-registration §14 (v));
- writes `resolvability_cells.parquet` (with `member`), `resolvability.parquet`
  (per-member rows, `member_decision` and the ensemble's `decision` rows with
  the rule, member statuses, spread and limit, saturated and filled flags),
  `resolvability_summary.json` (`resolvability_version` 7, members, `ensemble`
  statistics incl. the member spread and agreement, `emitted_before_fill`,
  the top-up, assets and chemistry, `profile_prediction`) and
  `resolvability_class_depth.parquet`.

`resolvability_class_depth.parquet` is the table RESOLVE will consume (M4
follow-up): one row per (regime, level, class, depth bin) with `status`,
`threshold` and `threshold_source` (`default`, `resolvability_local`,
`saturated_cap`, `monotone_inherited`), `ensemble_rule`, `pooled`,
`extrapolated`, `monotone_filled`, `nonneuronal_high_depth`, `neuronal`,
`n_test`, `n_confident` (distinct test cells), the ensemble `precision` and
`coverage` there (a filled bin: its fill measurement; a pooled bin: its
pool's), the member minimum and maximum of precision and coverage, and, when
the family's species x chemistry has a depth profile (the public 5K mouse
section), `profile_source`, `profile_share` s_c(d) and
`predicted_coverage_term` s_c(d) cov(L, c, d) on emitted bins. A dataset's
predicted coverage of (L, c) is the sum of `profile_share` x `coverage` over
emitted bins with its own s_c(d) (its cells called c), its resolvable share
the sum of s_c(d). `load_resolvability` refuses a version-7 bundle unless its
caller declares support (`allow_version_7=True`), so M4's RESOLVE refuses it
loudly until its follow-up (plan §12 M3c). The version-7 decisions of the
version-6 families are computed only as a diagnostic
(`annotation-panel-simulate --resolvability-version 7`, never written to the
store, never applied).

**Measured (M3c stage D, 2026-09-28; `m3c/` in the evidence archive,
`M3C_EXIT_REPORT.txt`).** Version-7 bundles were built as new build
directories of the large store (existing bundles untouched), 8 processes on
the shared host (load 35-70):

| Family | Bundles | Wall | Peak process-tree PSS | Provisional emission |
|---|---|---|---|---|
| Xenium Prime 5K human (5,001 genes; R1 x 3, clean and lung stress reported) | WHB `40886998`, SEA-AD `f0798063`, held-out test set `816f3afe` | 80.8 min (WHB 44.6 min, of which the self-map 35.2; SEA-AD 30.6 min) | 9.3 GB (largest process 7.8 GB) | WHB 312 of 546 bins (291 unanimous, 20 by the spread, 1 filled), SEA-AD 76 of 104 |
| Xenium Prime 5K mouse (5,006 genes; R1 x 3 + R3; clean reported) | WMB `e0590aac` (16.3 GB, 13.2 GB reference markers), test set `457e0140` (14,072 cells) | 6.0 h build (reference markers 63.5 min, query markers 37.7 min, test set with the top-up 2.8 min, five members 2.3-8.4 min thinning + 36-49 min mapping each, decisions 1 min, trust guard 0.8 min, tables 1.8 min); 7.6 h with profile mode | 25.8 GB over the run (reference markers 23.1 GB; largest process 21.3 GB, query markers) | 1,244 of 2,210 bins (1,052 unanimous, 179 by the spread, 13 filled, 4 at the saturated cap, 61 `nonneuronal_high_depth`) |

The human top-up had nothing to add: COP (152 test cells) and Fibroblast
(144) are the only classes below 200, and both pools (the held-out donor's
other cells, the other-region cells of training clusters) were exhausted.
The mouse top-up added 1,013 cells to 11 classes (OB-CR 60 -> 162, DG-IMN
70 -> 200, MH-LH 90 -> 200, MB Dopa 80 -> 200, MB-HB Sero 70 -> 168, CB
GABA 130 -> 200, HY MM and CB Glut 30 -> 200; HY Gnrh1 10 -> 17, Pineal
10 -> 15 and OEC 10 -> 41, whose pools ran out under the training-cluster
rule and the 5% cluster cap). A 5K member's exact-total thinning takes 2-8
min for 120-180k simulated cells (up to 16 min under load), its mapping 2-4 min (human) or 36-49 min (mouse, 8 processes at load
40-70). On the public 5K mouse section's own per-class depth the mouse
bundle's class-depth predictor gives provisional class coverage .909 and
subclass .786; profile mode (member mean, weighted to the real composition)
class .947 and subclass .789 against the real .871 and .695 under the same
decisions (the calibration of pre-registration §14 (iv), in-sample). A
fresh keyed ensemble (R1 at seeds 3-5, R3 at seed 1) moved 52 of the 1,263
provisional (level, class, bin) triples of the mouse bundle (churn 0.041,
of which the report-only supertype level 28; broad to subclass 0.022),
against 0.081 for two single R1 draws under the same conventions: the
pre-registered limit of 0.02 is not met (`M3C_EXIT_REPORT.txt` gives the
cause and the proposed fixes), while the real public-section coverage
under the two ensembles differs by only +.003 (class and subclass). The
failure stays on record. *Amendment (2026-09-29; orchestrator decisions
pending the user's confirmation, pre-registration §15):* eight emission
members and the one-standard-error margin on the spread route; the test is
re-run once on the rebuilt bundle against a comparator ensemble
(`R1_contam_HO@20`-`@25`, `R3_measured_HO@20`, `@21`), and the work stops if
it fails again.
Under the lung-FFPE depth scenario (median 245 counts) the human
profile mode predicts a provisional supercluster coverage of .593 (member
mean; phase 1's bracket .55-.57) and broad .776.

### Panel families, diagnostics and trust states (M3b)

**Validated families** are packaged in `assets/annotation/`:
`validated_panels.csv` (one row per validated panel hash: family, species,
platforms, gene count, `validated_max_level`, `validation_basis`
`real_data` | `simulation`, evidence, date, approving PR),
`validated_panel_genes.csv` (each row's resolved IDs and root markers) and
`validated_panel_levels.csv` (per (level, class) records of
simulation-validated families: status `validated` | `failed:NP<k>` |
`not_evaluable`, `validated_min_depth`, `tested_max_depth`; header-only until
the first gate-P PR, M13). The seeded families are all `real_data`:
`human_set_a` (the E5 296-gene set a, its post-M0e 297-gene form and their
curated set c, 264 / 265 genes; validated up to `supercluster`),
`mouse_ag7` (500) and `mouse_vzg2` (815; up to `subclass`).
`panel.validated_panels_path` in `annotation_config.json`
(`AnnotationPanelConfig`) points to another directory with the same files.

**Families** (`panel.panel_family`, OD-E7): a panel whose hash is a row of
the table for the same species and platforms is `listed` in its family; one
with Jaccard ≥ 0.95 to a row that contains all that row's root markers
`inherits` it; a subset panel keeps its parent's family; anything else is
its own family. A family counts when it was validated on every platform
the panel serves: one platform of a pair family run on its own
(`per_platform`, e.g. the P5011 Xenium panel, set a plus one gene) keeps
the family, while a Xenium panel never inherits a MERSCOPE-only family
such as VZG2. The family records the matched platforms
(`matched_platforms`).
`panel_report.json` records the tables' sha256 (`validated_panels`) and, per
annotation panel, `family_validation` (listed or not, basis and the trust
it is expected to get before PREP's checks).

**Diagnostics** (`annotation.diagnostics.panel_diagnostics`): gene-ID
resolution per declared panel (features in, controls removed by type,
genes by ID source, unmapped features, merged duplicates, species test)
and, per bundle, the panel coverage PREP recorded in `bundle.json`
(panel genes absent from the reference, root markers, root children with
≥ 10 markers, markers per parent, weak parents, auto-collapsed parents and
the leaves they hide) and the resolvability trust constraint.

**Trust states** (`annotation.diagnostics.trust_state`, per reference and
panel, first match wins; decided from the cached bundle tables, outside
`build_hash`, so promoting a family rebuilds nothing):

| State | Rule | Effect |
|---|---|---|
| `refused` | a declared panel refused by the gene-ID resolver; < 50 panel genes in the reference; < 10 root markers; broad unresolvable at every depth ≤ 250 | reference not mapped; primary: gate `failed` (`panel_refused`), every cell `not_attempted_gate` and `exclude_hard`; secondary: degraded mode `single_method` |
| `broad_only` | leaf resolvable for fewer than half of the classes with enough test cells at every depth ≤ 250; or the bundle of an unlisted panel has no self-map (fail-safe) | leaf and finer levels `not_attempted_gate`, `subcluster_status = not_resolvable_panel`, gate capped at `broad_only` |
| `validated` | family listed in `validated_panels.csv` | `real_data`: validated thresholds and packaged floors up to `validated_max_level`, `ct_<L>_validated` on every confident label there; `simulation`: emission exactly as provisional, `ct_<L>_validated` where (level, class) is validated at the cell's depth, gate warning only when > 10% of confident labels lie outside the validated region |
| `provisional` | anything else | provisional margins, local thresholds, max-rule floors (`unknown_panel`), banner and gate warning flag (never a lower gate level) |

Refused, broad-only and provisional panels show a report banner; validated
families never do. Measured on the current panels
(`m3b/review_fix/diagnostics/` in the evidence archive): the builder-v3
bundles of set a (WHB and SEA-AD, rebuilt at resolvability version 2), set c
(WHB `af201f26`, built in the review fix), ag7 and VZG2 all carry a self-map
and are `validated` (`real_data`) with no note and no resolvability
constraint. Only the older builder-v2 set-c bundle (`f20d11b0`) has no
self-map; it carries `resolvability_not_run`, and MAP no longer accepts it.
Of the P5011 per-platform panels, the Xenium one (298 genes: set a plus one
gene) inherits `human_set_a` and is `validated`, the MERSCOPE one (268
genes) is its own family and `provisional` (banner and gate warning, as H8
expects); ag7 symbols run as human are `refused`
(`gene_ids:species_mismatch`).

### Xenium Prime 5K panel card (M3c)

Panels of chemistry `xenium_prime` (Jaccard >= 0.95 with a pinned public
Prime 5K list) carry these panel-card notes
(`merxen.annotation.diagnostics.panel_card_notes`; printed by
`annotation-panel-simulate` and meant for the report's panel card, M7). They
state limits and never change a prediction, an emission or a trust state
(user decision 4 of 2026-09-28, plan §8.10):

- "Simulated glial coverage is an upper bound: -.06 to -.18 on
  vendor-segmented 5K cells, -.04 to -.14 re-segmented with ProSeg" and
  "Precision is unmeasured on real data" (the thinned-cell agreement, .996
  class / .987 subclass at 250 counts, is a self-consistency upper bound; the
  ProSeg range, added 2026-09-29, comes from a ProSeg re-segmentation of the
  public section with the vendor cells as its prior,
  `5k_real/phase1b/REPORT.txt`);
- trust: `provisional` with the provisional margins; nothing promotes the
  family automatically and the public 5K section never enters a gate or a
  promotion; gate P (M13) runs on the version-7 ensemble, with thresholds and
  emission frozen from the ensemble and NP3-NP7 required in every emission
  member;
- real datasets get the downgrade-only per-class coverage warning (a
  class's real coverage below its simulated class-depth prediction - 0.10;
  it fires for glia and small hypothalamic classes alike, and on v1-type
  large-mask segmentation); no empirical offset is applied;
- the 5K numbers come from one public section (one hemisphere, vendor XOA 3.0
  segmentation) and are in-sample for depth, composition and factors;
- mouse: before any gate-P PR, the first in-house dataset (MerXen
  segmentation) measures per-class depth, factors and contamination, and PREP
  is re-run with them as new simulation inputs, never as trust evidence;
- human: glial coverage is expected below simulation by analogy with mouse;
  there is no R3 member (no factor table against WHB) and no mouse factor,
  depth profile or depth prior; predictions are per depth scenario (per grid
  bin and the lung-FFPE scenario, median 245 counts); the family stays
  provisional until an in-house human 5K dataset measures per-class depth,
  gene complexity, factors against WHB and real vs simulated coverage.

### Real-data QC (M3c; downgrade-only)

`merxen.annotation.real_qc` holds the checks M3c adds to the label-free QC
of real datasets (plan §8.8). They are pure functions (tables in, records
out) that can only warn or report; RESOLVE wires them per dataset in the M4
follow-up and M13 wires the first in-house dataset of a family. None changes
emission, a threshold, a floor, a label or a trust state, and
`apply_qc_outcomes` combines outcomes with a trust state so that it can only
stay or fall (property-tested).

| Check | Function | Rule | Effect |
|---|---|---|---|
| Per-class real vs simulated coverage (version-7 families) | `class_bin_shares`, `predicted_class_coverage`, `real_class_coverage`, `coverage_vs_simulation` | per (level, called class) with >= 200 cells: the dataset's confident share against `sum_d s_c(d) cov(L, c, d)` from `resolvability_class_depth.parquet` at the dataset's own per-class bin shares (the pre-registered predictor); where profile mode has run, its per-class prediction (`profile_coverage_table`) is reported beside it (`profile_coverage`) and never decides | warning per class when real < simulated - 0.10, worded per class (it fires for glia and for small hypothalamic classes alike); the text says the warning also fires on v1-type large-mask (nucleus-expansion) segmentation, where simulation over-predicts coverage by +.16 to +.22 |
| Non-neuronal high depth | `nonneuronal_high_depth_flags` | non-neuronal cells at >= 1,000 counts in emitted `nonneuronal_high_depth` bins | report-only `flag_nonneuronal_high_depth` |
| Glial large-mask / high-depth trend | `nonneuronal_depth_trend` | real non-neuronal coverage at >= 1,000 counts below the 500-999 band by more than 2 SE (both >= 200 cells) | report-only (5K vendor glia: class .916 -> .897 -> .886) |
| Factor re-measure (first in-house dataset of a family with a measured factor table) | `factor_remeasure` | per-gene factors re-measured with the X1 code (`shadow.reference_pseudobulk_totals`) on the confident calls, centred on the median informative gene and capped +-3, against the stored table on genes informative in both | warning when Pearson r < 0.9, recommending a PREP re-run with the in-house table as a new asset (not automatic) |
| Gene complexity | `gene_complexity_check` | median genes per cell of native vs simulated cells per depth bin (>= 50 cells each) | warning when native cells carry > 45% more genes; its text says simulated coverage predictions are unreliable for the dataset |

The thresholds are the `real_qc` fields of the annotation config
([Configuration](../configuration.md)).

## Pipeline processes

Three CPU processes in `workflows/modules/annotation.nf`, wired by
`workflows/subworkflows/annotation_references.nf` (PANEL, PREP) and
`workflows/subworkflows/clustering_map_first.nf` (MAP); none takes the GPU
lock, and a default (legacy) run instantiates none of them.

| Process | Runs | Resources | What it does |
|---|---|---|---|
| `ANNOTATE_PANEL` | once per pair × segmentation | 1 CPU, 4 GB | `merxen annotation-panel` on the gene list (`--annotation_panel_genes_path`) or on the pair's prepared H5ADs; writes the declared panels and `required_bundles.json`. |
| `ANNOTATE_REFERENCE_PREP` | once per unique (species, reference, `panel_hash`) across the run | 8 CPUs, 64 GB, 8 h; above 1,000 panel genes `annotation_prep_large_memory` and 24 h; one at a time on dwight | `merxen annotation-reference-prep`: gets the bundle from the store or builds it, and writes `bundle_ref.json`. Seconds when the bundle exists. |
| `CLUSTERING_SQUIDPY_ANNOTATE_MAP` | once per pair × segmentation, after its last required bundle (`map_first` only, from M5) | 6 CPUs, 24 GB (48 GB above 1,000 panel genes); `annotation_max_forks` (2) at a time on dwight | `merxen annotate` on the pair's prepared H5ADs with exactly the bundle refs PREP resolved: the MapMyCells runs, the tidy parquets, the provisional labels and `map_manifest.json`, published to `<outdir>/<pair>/<seg>/annotation_map/annotation_map_out/`. |

PREP has no `storeDir`: the store's own lock, temporary build directory and
atomic rename keep concurrent launches safe, and its `build_hash` (sources,
`cell_type_mapper` version, builder settings) is only known inside the task.
PREP is never cached (`cache false`): the task hash cannot see the builder
code or the content of its source files, so a cached task could hand MAP a
stale bundle after a builder fix. A re-run takes seconds when the bundle
exists, and `bundle_ref.json` holds only the bundle's identity (whether it
was reused is logged), so an unchanged bundle gives byte-identical output
and MAP's cache holds (MAP caches on file content, `cache "deep"`). Which
pair's copy of a shared panel file reaches PREP first does not matter: the
bundle depends only on the panel's IDs.
Each pair × segmentation is then released with exactly the bundle refs its
`required_bundles.json` lists (a same-panel human pair on `proseg_hybrid`
needs three: WHB and SEA-AD on set a, WHB on set c; a `per_platform` pair
five; a mouse section two), as soon as its last bundle is ready, so pairs
with different panels never wait for each other. A pair whose panels are
refused is released with no bundle; one whose PREP failed is dropped.

MAP then runs with 6 MapMyCells worker processes (`--n-processors`
`task.cpus`, one BLAS / numba thread each, `CUDA_VISIBLE_DEVICES` empty, this
checkout's `src/` first on `PYTHONPATH`) after checking that the installed
`cell_type_mapper` is `annotation_ctm_version`. Its table cells and
`min_counts` come from the clustering config, so they are the clustering
run's. A refused panel is not a task failure: MAP writes a `map_manifest.json`
with `panel_status: refused` and its reasons, maps nothing, and RESOLVE (M4)
will write statuses only. With `annotation_reuse_published` a run whose
query fingerprint, `build_hash`, engine parameters, ctm version, tidy schema
version and (restricted) lookup equal the published manifest's is copied
from `annotation_map/annotation_map_out/` instead of re-mapped, because
dwight prunes work directories; with `annotation_keep_extended_json` the
published run must have kept its JSON too. A published manifest that cannot
be read (the `-stub-run` manifest, an older or newer layout) only disables
reuse, with a warning. Until M5 wires
`map_first` (hook H5, `CLUSTERING_MAP_FIRST`), the preflight refuses
`map_first` runs, so MAP runs only in the workflow tests; the shadow
evaluation uses the standalone command.

### Pre-building references (`--annotation_prepare_only`)

```bash
nextflow run workflows/main.nf \
    --samplesheet samplesheet.csv \
    --annotation_prepare_only true \
    --annotation_panel_genes_path set_a_genes.csv
```

The run builds the bundles of the declared panel in the gene list (any file
`merxen annotation-panel --panel-genes-path` reads) for every samplesheet row
and analysis segmentation, and runs no pipeline stage: rows are neither
preflighted for their stages nor built, segmented or clustered. The species
comes from `--species` and the references from
`annotation_<species>_references`; SEA-AD and the MERFISH CCF metadata are
fetched once into `<annotation_reference_store>/.downloads` (pinned URL,
size and sha256; `annotation_auto_download`). Its own preflight checks the
gene list, the references' source params, the region, the self-map sources
while `annotation_resolvability` is true (human WHB / SEA-AD:
`annotation_whb_region_precompute_source`, `annotation_whb_h5ad_dir` and
`annotation_whb_metadata_dir`; mouse `wmb_panel`: the self-map test cells)
and that the stores are writable. A later `map_first` run reuses a bundle whenever its declared
panel resolves to the same Ensembl IDs (`panel_hash`), whatever symbols the
gene list or the platforms declare. A human gene list of the seeded set-a
family also gets its curated set c, so a prepare-only run builds all three
bundles a same-panel pair on `proseg_hybrid` needs. Building from a pair's
prepared H5ADs (`per_platform` panels, label-free set c) arrives with the
`map_first` wiring (M5). The params are listed in
[Configuration](../configuration.md#reference-based-annotation-in-development).

## Mapping (`merxen annotate`)

The MAP step (`merxen.annotation.pipeline.annotate_map`; plan §3.3) maps
each sample of a pair × segmentation onto every use of a bundle its
`required_bundles.json` lists with a primary or secondary role: WHB and
SEA-AD on the sample's annotation panel, WHB on set c for the segmentations
in `annotation_xplat_sensitivity_segmentations` (`proseg_hybrid` by
default), and WHB on the intersection panel for `per_platform` pairs. A
bundle is listed once per (reference, panel hash) with every purpose in
`uses`; `map_bundles` makes one run per use, on that use's panel file (whose
platforms decide which samples it maps) and with that purpose's run id. When
two uses share a gene set (a small MERSCOPE panel inside a Xenium panel
makes the intersection equal the MERSCOPE panel; set c can equal set a) the
query is mapped once and recorded under both run ids (`same_mapping_as`). A
sample that no run applies to fails the task, unless its own platform panel
was refused (`panel_status: refused` for that sample). The standalone
command runs it on published clustered H5ADs (options in
[CLI](../cli.md#merxen-annotate)); `CLUSTERING_SQUIDPY_ANNOTATE_MAP` runs it
on the prepared H5ADs of a `map_first` run (see
[Pipeline processes](#pipeline-processes)).

Per sample:

1. Load the counts (prepared `X`, or the published `layers["counts"]`),
   remove control features with the shared registry (the same features
   legacy `remove_control_features` removes on the current panels) and take
   `total_counts` / `n_genes` from `select_table_cells`. Gene IDs resolve as
   in `annotation-panel` (the resolver above; each feature keeps its
   declared decision).
2. Map the table cells (`total_counts >= min_counts`; a published clustered
   H5AD holds only table cells) on the panel's genes present in the dataset.
   Missing panel genes are recorded; a missing marker gene restricts the
   bundle's lookup (parents left without markers are auto-collapsed). A
   **subset bundle** is needed when a missing gene is a root marker, a
   parent is left with fewer than 5 markers (`weak_parent_markers`) or more
   than 1% of the panel is missing (`subset_bundle_missing_frac`); above 5%
   (`own_family_missing_frac`) the subset is its own panel family, otherwise
   it keeps the parent's. MAP then writes the subset panel
   (`subset_panels/<sid>_<run_id>.panel_genes.json`). A standalone
   `merxen annotate` with `--store` maps with the store's bundle on it if
   there is exactly one (opened as a bundle ref would be; several are
   recorded as `ambiguous`); a pipeline task (`--require-bundle-refs`)
   never looks in the store, because a bundle PREP did not stage is
   invisible to `-resume`. Otherwise MAP maps with the restricted lookup
   and records the request (`subset_bundle` in the run record). Build the
   requested bundle with `merxen annotation-reference-prep --panel-genes
   <subset panel>`. Of several builder-v3 bundles on one panel, a
   standalone run takes the one built with the large-panel prefilter its
   config asks for and, of those, the one with the current
   `RESOLVABILITY_VERSION`. When none has the current version (a panel not
   rebuilt since the last bump), it takes the older bundle and logs a
   warning naming its resolvability version: its self-map tables are stale
   until the panel is rebuilt (a pipeline run's PREP rebuilds it).
3. Run MapMyCells (seed 0, bootstrap factor 0.5, 100 iterations, raw
   normalisation, one BLAS thread per worker, `--drop_level
   CCN20230722_SUPT` for WMB) and parse the extended JSON at once into the
   tidy parquet; the JSON is deleted unless `annotation_keep_extended_json`.
   Mouse maps unpruned for now: region inference and the pruned re-map are
   M6.
4. Write `map_manifest.json`: per sample the input identity, table-cell
   counts, controls removed and, per run, the query fingerprint (sha256 of
   the cell ids, their total counts and the query gene IDs), the bundle's
   `build_hash` and lookup digest, the engine parameters, the ctm version and
   commit, the settings the extended JSON recorded, wall time and peak RSS.

| File | Content |
|---|---|
| `<platform>/<sid>_mmc_<run_id>.parquet` | One row per mapped cell × taxonomy level; `run_id` is the reference id, `+_setc` for set c, `+_xpanel` for the intersection run of a `per_platform` pair. Run metadata in the parquet schema (`merxen_mmc`). |
| `<platform>/<sid>_ct_provisional.parquet` | One row per object: identity, `total_counts`, `n_genes`, `in_table`, **provisional** `ct_<level>_{name,raw,corr,runner_up,margin,status}` and `ct_final_*`, and the raw engine columns `mmc_<reference>_<level>_{label,name,bp,agg,corr}`. |
| `map_manifest.json` | The run record above, with `panel_status` (`ok`, or `refused` with `panel_reasons` and no runs) and, per run that needs one, `subset_bundle` (trigger, `used`, `requested` or `ambiguous` with the candidates, subset and parent hashes); `annotation-store prune` counts its `build_hash` values as references. |
| `subset_panels/<sid>_<run_id>.panel_genes.json` | The subset panel of a run whose dataset lacks enough panel genes (`kind` `subset`, `parent_panel_hash`, `excluded_ids`). |

The provisional labels apply the raw thresholds only (WHB lineage / broad /
NT 0.73 on the bootstrap probability summed over the assigned node's class,
supercluster 0.69, SEA-AD subclass 0.55 below `second_vote_below_counts`
(60) and 0.45 from it on the subclass `aggregate_probability`, E2's
definition; WMB class 0.90, subclass 0.80; probabilities are stored as
float32, so a threshold test allows 1e-6 below the threshold), the hard floor (`min_counts`), sinks and
frontal-cortex plausibility from the bundle vocabulary and the parent
chain. They have no floors, resolvability, dataset gate, second vote or COP
rule and are for inspection only; the RESOLVE step (M4) replaces them and
writes `<sid>_celltype_labels.parquet`.

A published clustered H5AD of a small sample can have a `min_cells`-filtered
`var` (P1212 and P5011 reseg MERSCOPE hold 299 and 268 of 300 features).
Its declared panel is the control-filter record, so it keeps the prepared
panel's hash and bundles; the filtered-out features have no native ID there
and resolve by symbol (pair lookup, local gene table). MAP maps the genes
present and records the missing ones (a subset bundle when they matter).
A clustered H5AD without the record (written before M1) declares its `var`;
give it the declared panel with `merxen annotation-panel --panel-file
<PLATFORM>=<declared panel file>` and `--panel-dir`.

## Shadow baselines (M3)

`scripts/acceptance/shadow_baselines.py` scores `merxen annotate` outputs of
published human datasets with `merxen.annotation.shadow` (plan §12 M3 item
1). It computes soft / argmax / confident broad JSD MERSCOPE vs Xenium with
spatial block-bootstrap CIs (500 µm tiles, 200 replicates; whole section and
shared tissue mask), a shadow evaluation of the v1 human rules (§5.2 lineage,
broad with the COP rule, supercluster; packaged floors; the dataset gate;
second-vote variants, including E2's likelihood-typer rule), WHB–SEA-AD
agreement, implausible and COP shares, the E1 marker referee and the M3 exit
check against the pilot. The measured baselines and the pre-registered
thresholds are in
[docs/acceptance/annotation-v1-preregistration.md](../acceptance/annotation-v1-preregistration.md).
Baselines may only tighten a threshold.

The other shadow items (plan §12 M3 items 2–7) have their own scripts, each
reading the `merxen annotate` outputs and the published inputs read-only:

| Script | Item | What it does |
|---|---|---|
| `scripts/acceptance/shadow_e8.py` | 2, E8 | Confident cells per mm², foreign-marker fraction and platform JSD for the four segmentations of a human pair and a mouse section (OD-B6, OD-B7) |
| `scripts/acceptance/shadow_x1.py` | 3, X1 | Reference-pseudobulk per-gene factors (±2 log2 cap), a rescaled WHB re-map, JSD with paired CIs and the referee vs set a and set c |
| `scripts/acceptance/heldout_genes.py` | 4, H4 | Held-out markers removed from query and lookup, a WHB-only re-map, fold enrichment and AUROC per class and platform (variants set a, set c, X1) |
| `scripts/acceptance/shadow_flags.py` | 5, H16 | Prototype contamination (dataset-empirical beta-binomial null) and diffuse-profile (multinomial q95) flags; realised rates per class × platform |
| `scripts/acceptance/shadow_ll.py` | 6, OD-B8 / OD-B13 | LL (vii) on every table cell; coverage and referee outcomes with and without the LL vote |
| `scripts/acceptance/shadow_glial_jsd.py` | 7 | WHB vs SEA-AD glial JSD with a paired block-bootstrap CI |

The results and the decisions they feed (X1, OD-B6 / OD-B7, OD-B8, OD-B13, the
H4 and H16 baselines) are in §11 of the pre-registration document.

## Known limitations

- **Four WMB subclasses have no 10Xv3 reference cell** (`157 RN Spp1 Glut`,
  `279 PSV Pax2 Gly-Gaba`, `280 NLL-po Pax7 Gaba`, `297 CU-ECU Pax2 Gly-Gaba`;
  sequenced only by 10X Multiome, ≤ 0.2% of cells in any in-scope posterior
  section). They are not filled in, but they stay in the mapping tree and
  the Allen means, so `cell_type_mapper` can still assign them with their
  ancestors' markers (the validated runs did: 12 ag7 cells as
  `157 RN Spp1 Glut`). Such labels have no marker support of their own:
  each mouse bundle lists these nodes in `marker_unsupported_nodes` (and
  `uncovered_subclasses`, `uncovered_clusters`), and RESOLVE reports them as
  unresolved at that level.
- **A fifth subclass, `261 HB Calcb Chol`, is missing from the marker build**
  because all 19 of its 10Xv3 cells are self-map test cells, which stay out
  of the build so the test set is disjoint (as are 4 other clusters whose only
  sampled cells are test cells). It is still in the mapping tree (Allen
  means), is auto-collapsed in the lookup and is listed with the four above:
  every `wmb_panel` bundle on ag7 and VZG2 reports 5 uncovered subclasses and
  19 uncovered clusters.
- Auto-collapsed parents are still assigned inside by `cell_type_mapper`
  (with their ancestors' markers); the nodes below them are in
  `marker_unsupported_nodes` and must not be emitted.
- **Only ag7, VZG2 and panels inside their gene union use the validated
  mouse configuration.** A new mouse panel uses its own genes as the marker
  universe (not validated; `validated_configuration.marker_universe` false)
  until M3's shadow run repeats the MO1 / E3 marker-consistency checks on
  it. Before this was fixed the pipeline built ag7 and VZG2 on their own
  genes too, and their lookups differed from the validated ones in 301 and
  142 of 368 parents (median Jaccard 0.979 and 0.992).
- **H18 does not pass as written on the validated panels** (resolvability
  version 6, M3b review 3, 2026-09-28; a decision for the gate PRs;
  `m3b/final_followup/FINAL_FOLLOWUP.txt` §12,
  `m3b/final_followup/review3/H18_V6.txt`, `EXCEPTIONS_V6.txt`). Version 6
  is the outcome of the two requested changes (other-region cells only of
  training clusters, per-cell keyed draws) with the pre-registered
  seed-0 efficiency. Set a (PREP) fails broad Oligo at 120 counts (79
  calls, precision .911, Wilson bound .828 < .88; 8.9% of the confident
  weight is other-region COP cells called Oligodendrocyte); version 4
  failed broad Astro and broad and supercluster Oligo at 120, and
  supercluster Astro at 120. Reweighted to the eight human datasets 23
  (level, class) pairs fail (version 4: 25): broad OPC on all eight (at 15
  counts: precision .83-.90, held-out-donor neurons and Oligodendrocytes
  called OPC; at 120 on five: precision .90-.94 but Kish n 88-90, so the
  Wilson bound misses), broad Oligo on seven (at 120 counts on five, at 15
  on P7513 MERSCOPE, at 15-120 on both P5011) and supercluster Oligo on six
  (at 120 on four, at 30-120 on both P5011; other-region COP cells of
  training clusters called Oligodendrocyte hold 9-26% of the called set's
  confident weight), and broad and supercluster Immune at 60
  on P5011 MERSCOPE (held-out-donor deep-layer IT neurons called Immune,
  7-10%). COP supercluster is never emitted. Mouse: VZG2 passes; ag7 PREP
  fails the Immune subclass at 100 (113 calls, precision .894, Wilson .824
  < .83; as in version 4; it passes with seeds 1 and 2); reweighted, ag7
  fails 05 OB-IMN GABA (at most 3 positive-weight confident calls at 250
  counts on proseg_hybrid and reseg; Kish n 34-82 on original_seg and
  proseg_mask) and the 09 CNU-LGE GABA subclass (precision .92, Kish n
  58-64): 1 / 2, 1 / 1, 0 / 1, 0 / 2 classes / subclasses on proseg_hybrid,
  reseg, original_seg, proseg_mask (version 4: 1 / 1, 1 / 1, 0 / 0, 0 / 0).
  Validated thresholds the local rule would raise (fit half ≥ 50 calls, at
  D ≥ 15 broad, 30 supercluster, 20 class, 50 subclass): set a 19 (PREP;
  12-23 reweighted), SEA-AD 8, set c 6, ag7 39, VZG2 68; 0, 0, 5, 94 and
  104 bins have no fit (`validated_thresholds_not_evaluable`).
- **The H7 coverage proxy misses on P5011 MERSCOPE under version 6.** The
  confident broad share of table cells after reweighting (validated
  regime; it ignores the SEA-AD vote, the COP rule on real cells and the
  floors, so the real H7 values will be lower) is, version 4 → 6: P7513
  MERSCOPE .683 → .683 (H7 asks ≥ .62), P7113 MERSCOPE .739 → .739 (≥
  .67), P1212 MERSCOPE .354 → .354 (≥ .34), P5011 MERSCOPE .297 → .298
  (≥ .30: fails); Xenium P7513 .610 → .605, P7113 .662 → .656, P1212 .528
  → .523, P5011 .546 → .557 (≥ .44 / .39 / .29 / .40). Version 5's pass
  (P5011 MERSCOPE .380) came from its unrequested redraw of the gene
  efficiency, not from the two changes.
- **The H18 / H7 verdicts depend on the simulation draw** (version 6,
  `m3b/final_followup/review3/FACTORIAL_ANALYSIS.txt`). A 3 × 3 grid on
  set a crossed the per-cell seed (thinning, spill, partner) with the
  efficiency seed (one realisation per cell, mapped at seed 0; the
  additive model's residual holds the interaction and the mapping noise).
  Across it P5011 MERSCOPE's H7 proxy spans .297-.431 (below .30 in 2 of 9
  cells, both with the seed-0 efficiency), P1212 MERSCOPE .320-.444 (below
  .34 in 1), the others stay ≥ .028 above target; the reweighted H18
  failures number 12-26 and set a PREP's 2-8 (both floor sets). Changing
  only the efficiency seed moves on average 6.3 of the 83 validated broad /
  supercluster bins per dataset (1-14), only the per-cell seed 4.4 (0-9),
  both 6.5; mapping the same simulated cells in another order (the
  MapMyCells per-chunk bootstrap) 1.4 (0-2), and that alone moves P1212
  MERSCOPE's proxy from .354 to .433 and adds 4 set a PREP failures. For
  the H7 proxy and the H18 counts neither factor is significant in this
  grid (F(2, 4) < 6.94, except P1212 Xenium's H7 proxy: efficiency, range
  .007), and the residual is the largest component for most metrics; only
  broad OPC at 15 clearly follows the efficiency draw (it fails on all
  eight datasets with the seed-0 efficiency under every per-cell seed and
  on 0-5 with seeds 1 and 2; F = 18.1, p = .01). How the gate should treat
  this spread (the pre-registered seed-0 draw, several draws, or the worst
  of them) is open for the user; no verdict is recorded as a pass meanwhile.
- **Simulated coverage on Xenium Prime 5K is optimistic for glia, and one
  efficiency draw decides emission** (phase 1, 2026-09-28;
  `5k_real/SYNTHESIS.txt` §0, §2, §4.4, `5k_real/sim/REPORT.txt` §6). At the
  real per-class depth, simulation exceeded real vendor-segmented coverage
  by +.04 to +.09 overall, +.06 to +.18 for glia (Astro-Epen, OPC-Oligo,
  Vascular, Immune) and +.09 to +.23 for small hypothalamic classes, in-sample
  on one section; about 80% of the glial excess is not explained by mask
  size. Two draws of the same recipe moved 47 of ~450 emitted (level, class,
  bin) triples of the 5K mouse bundle. Precision is not measured on real
  data. Both 5K families stay `provisional`. Planned responses (plan §8.3
  resolvability version 7, §8.8, §8.10; milestone M3c): per-class depth
  profiles, a draw ensemble with the measured 5K factors as one member,
  panel-card notes and a downgrade-only per-class coverage warning; no
  empirical offset. Set a, ag7 and VZG2 keep resolvability version 6.
  *Implemented in M3c (2026-09-28):* the version-7 bundles of both 5K
  families exist; M4's RESOLVE refuses them until its follow-up (plan §12
  M3c), and the real-data QC functions (`real_qc`) are not yet wired into
  RESOLVE or the report. On the public section the version-7 predictions
  still exceed the real coverage by +.07 (class) to +.09 (subclass) overall
  and by up to +.25 for hypothalamic classes, and the per-class warning
  fires for 19 (level, class) pairs (`M3C_EXIT_REPORT.txt`).
- **Version-7 cost on 5K panels:** exact-total thinning of 5,000 genes takes
  2-8 min per member, and a 5K mouse member maps in 36-49 min on 8 processes
  on the shared host, so a version-7 5K mouse PREP took about 6 h in stage D
  (four emission members and the clean bound) and its fresh-ensemble
  diagnostic another 4 h; with the eight emission members of the amendment
  of 2026-09-29 the estimates are about 9-10 h and 8 h.
- **Pooled depth scenarios:** a pooled scenario (the lung-FFPE totals of
  human Prime) has no class composition, so `annotation-panel-simulate`
  weights its per-level class-depth headline and profile mode's ALL row by
  the test-set composition (each class's share of the self-map test cells),
  labelled "weighted to the test-set composition" (since 2026-09-29); the
  per-class predictions are in `<reference>/profile_class_depth.csv` and
  `profile_predictions.csv`.
- **Set c of families without a curated list** uses the label-free rule,
  which drops far more genes than E5's validated set c (44-73 per pair on the
  E5 pairs); treat such set-c results as provisional.

## Licences

ABC Atlas, WHB, WMB, MERFISH-C57BL6J-638850 and SEA-AD data are CC BY-NC 4.0;
`cell_type_mapper` is under the Allen Institute Software License.
