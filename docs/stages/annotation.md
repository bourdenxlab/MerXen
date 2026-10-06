# Reference-based annotation (in development)

Reference-based cell-type annotation replaces the legacy marker scoring of
[Squidpy clustering](clustering-squidpy.md) when a species runs in
`map_first` mode. It is being built milestone by milestone
(`docs/plans/robust-celltype-annotation-plan.md`); legacy runs are
unchanged. This page covers what exists so far: declared panels
(`merxen annotation-panel`), the reference bundles that
`merxen annotation-reference-prep` builds into the reference store, the
two pipeline processes that run them (`--annotation_prepare_only`), the
MAP step (`merxen annotate` and its pipeline process
`CLUSTERING_SQUIDPY_ANNOTATE_MAP`), which maps samples onto the bundles, the
RESOLVE step (label tables) and, since M5, `map_first` pipeline runs that
build the map-first hierarchy from the labels
([Map-first clustering runs](#map-first-clustering-runs-m5)). Human runs
map_first by default since its flip (M8, pre-registration §20); mouse stays
legacy until its flip (M9).

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
these nodes with their ancestors' markers, so mouse RESOLVE (M6) reports
such a label as `not_resolvable` at that level; human RESOLVE does not apply
this rule yet (pending). The nodes stay in the mapping tree, as in the
validated runs.

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
   `bundle.json`). *Other-region COP cells (user decision 2026-09-30, M8
   D1; pre-registration §18):* every human self-map on
   `whb_frontal_supc_clus_ho` leaves out the test cells whose `test_source`
   is `other_region` and whose truth supercluster is Committed
   oligodendrocyte precursor (set a: 132 of its 152 COP test cells; the 20
   held-out-donor COP cells stay), because almost every wrong broad
   Oligodendrocyte call of the self-map was such a cell. The held-out bundle
   and its `test_cells.h5ad` are unchanged: `reference.self_map_test_cells`
   leaves the cells out when the self-map loads them, and the rest is
   simulated with the keyed draws. The rule, its revision
   (`HO_SELF_MAP_TEST_SET_REVISION` 1), the test source and the excluded
   superclusters are hashed (`test_set.self_map_exclusion` of the build
   params), so the rebuilt WHB set a / set c and SEA-AD set a bundles have
   new build hashes at the same `RESOLVABILITY_VERSION` 6;
   `resolvability_summary.json` and `bundle.json` record
   `test_set_exclusion` (revision, rule, cells left out per supercluster,
   kept cells of those superclusters per source). The approval covers the
   M8 bundles; applying it to later human families on this test set (5K,
   the M13 panels) needs the user's confirmation.
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
| `resolvability_summary.json` | Recipes, levels, settings, test-set counts, `D_max` per level and class, emission per regime, level, class and depth (status, threshold, `t*`, extrapolated, pooled, reason), the pooled deep sets per regime, level and class (`D_P`, the bins taking their verdict, statistics, `t*`, status), floors, validated thresholds the local rule would raise (own bins whose fit half holds 50 calls; pooled sets carry `would_raise`), the validated bins whose fit half holds fewer (`validated_thresholds_not_evaluable`: `t*` unknown, never counted as a raise), the trust constraint, fine-level seed stability, runtimes and mapping runs (with each run's `n_processors`); human held-out self-maps: `test_set_exclusion` (the M8 D1 revision, rule and the cells left out). |
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
  confirmation; plan §8.3 v7.3, pre-registration §22) -- `R1_contam_HO@0`,
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
  (`trust_asset_guard` in the summary; pre-registration §21 (v));
- writes `resolvability_cells.parquet` (with `member`), `resolvability.parquet`
  (per-member rows, `member_decision` and the ensemble's `decision` rows with
  the rule, member statuses, spread and limit, saturated and filled flags),
  `resolvability_summary.json` (`resolvability_version` 7, members, `ensemble`
  statistics incl. the member spread and agreement, `emitted_before_fill`,
  the top-up, assets and chemistry, `profile_prediction`) and
  `resolvability_class_depth.parquet`.

`resolvability_class_depth.parquet` is the table RESOLVE consumes (the M3c
follow-up; RESOLVE derives it from the decisions it applies, reweighted to the
dataset, see "Resolvability version 7 in RESOLVE"): one row per (regime,
level, class, depth bin) with `status`,
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
caller declares support (`allow_version_7=True`), so a consumer written for
version 6 only refuses it loudly; RESOLVE (human and mouse) and
`annotation-panel-simulate` declare it. The version-7 decisions of the
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
decisions (the calibration of pre-registration §21 (iv), in-sample). A
fresh keyed ensemble (R1 at seeds 3-5, R3 at seed 1) moved 52 of the 1,263
provisional (level, class, bin) triples of the mouse bundle (churn 0.041,
of which the report-only supertype level 28; broad to subclass 0.022),
against 0.081 for two single R1 draws under the same conventions: the
pre-registered limit of 0.02 is not met (`M3C_EXIT_REPORT.txt` gives the
cause and the proposed fixes), while the real public-section coverage
under the two ensembles differs by only +.003 (class and subclass). The
failure stays on record. *Amendment (2026-09-29; orchestrator decisions
pending the user's confirmation, pre-registration §22):* eight emission
members and the one-standard-error margin on the spread route; the test is
re-run once on the rebuilt bundle against a comparator ensemble
(`R1_contam_HO@20`-`@25`, `R3_measured_HO@20`, `@21`), and the work stops if
it fails again.
Under the lung-FFPE depth scenario (median 245 counts) the human
profile mode predicts a provisional supercluster coverage of .593 (member
mean; phase 1's bracket .55-.57) and broad .776.

**Measured after the amendment (M3c stage E2, 2026-09-29/30; code
`6bb42fd`; `M3C_EXIT_REPORT.txt` §10).** The amended code (eight emission
members, the one-SE spread margin) was built into new build directories
of the large store; the stage-D bundles above stay in the store unchanged
as history. Host load 6-92, 8 processes per job:

| Family | Bundles | Wall | Peak process-tree PSS | Provisional emission |
|---|---|---|---|---|
| Xenium Prime 5K human (R1 x 8; clean and lung stress reported) | WHB `7e881fb1`, SEA-AD `1306b298`, held-out test set `816f3afe` (reused) | 2.5 h (WHB 68 min, SEA-AD 69 min) | 9.6 GB | WHB 310 of 546 bins (285 unanimous, 24 by the spread, 1 filled; 1 margin failure), SEA-AD 75 of 104 |
| Xenium Prime 5K mouse (R1 x 6 + R3 x 2; clean reported) | WMB `f6127077` (16.3 GB), test set `457e0140` (reused, 14,072 cells) | 9.1 h (reference markers 76 min, query markers 64 min, self-map 6.6 h: nine members at 3-6 min thinning + 26-58 min mapping each, decisions 4.5 min, trust guard 3.1 min, tables 4.4 min) | 28.9 GB (reference markers; largest process 21.3 GB, query markers) | 1,197 of 2,210 bins (1,014 unanimous, 174 by the spread, 9 filled, 2 at the saturated cap; 41 spread and 29 margin failures) |

All three are `provisional` (trust constraint none; for the mouse bundle
the asset-free guard was applied). The re-test of the emitted-triple
stability (pre-registration §22.4) **failed**: against a comparator
ensemble (`R1_contam_HO@20`-`@25`, `R3_measured_HO@20`, `@21`; 7.9 h, 21.1
GB) 62 of the 1,235 provisional triples moved (churn 0.050 > 0.02; broad to
subclass 0.031; supertype 28 of the 62; two single draws 0.060). In the
ensemble that did not emit a churned triple the member-spread limit was the
reason for 32 of them and the new margin for 13. The work stopped there, as
pre-registered. On the public section the two ensembles give the same real
provisional class coverage (.873 / .871) but subclass .643 / .687, mostly
through OPC-Oligo subclass (.564 / .824, three bins at 250-500 counts). The
mouse class-depth predictor on the section's own per-class depth gives
provisional class .909 and subclass .735, profile mode (member mean,
weighted to the real composition) class .945 and subclass .733. Under the
lung-FFPE scenario, weighted to the test-set composition, the human profile
mode predicts provisional broad .776 and supercluster .575 (members
.545-.593).

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
out) that can only warn or report. RESOLVE writes `flag_nonneuronal_high_depth`
and the class-depth prediction for version-7 bundles (the M3c follow-up); M13
wires the warnings into RESOLVE and the first in-house dataset of a family.
None changes
emission, a threshold, a floor, a label or a trust state, and
`apply_qc_outcomes` combines outcomes with a trust state so that it can only
stay or fall (property-tested).

| Check | Function | Rule | Effect |
|---|---|---|---|
| Per-class real vs simulated coverage (version-7 families) | `class_bin_shares`, `predicted_class_coverage`, `dataset_class_depth_prediction` (RESOLVE's predictor; label-free depth for classes with fewer than 100 cells), `real_class_coverage`, `coverage_vs_simulation` | per (level, called class) with >= 200 cells: the dataset's confident share against `sum_d s_c(d) cov(L, c, d)` from `resolvability_class_depth.parquet` at the dataset's own per-class bin shares (the pre-registered predictor); where profile mode has run, its per-class prediction (`profile_coverage_table`) is reported beside it (`profile_coverage`) and never decides | warning per class when real < simulated - 0.10, worded per class (it fires for glia and for small hypothalamic classes alike); the text says the warning also fires on v1-type large-mask (nucleus-expansion) segmentation, where simulation over-predicts coverage by +.16 to +.22 |
| Non-neuronal high depth | `nonneuronal_high_depth_flags` | non-neuronal cells at >= 1,000 counts in emitted `nonneuronal_high_depth` bins | report-only `flag_nonneuronal_high_depth`, written by RESOLVE for version-7 bundles |
| Glial large-mask / high-depth trend | `nonneuronal_depth_trend` | real non-neuronal coverage at >= 1,000 counts below the 500-999 band by more than 2 SE (both >= 200 cells) | report-only (5K vendor glia: class .916 -> .897 -> .886) |
| Factor re-measure (first in-house dataset of a family with a measured factor table) | `factor_remeasure` | per-gene factors re-measured with the X1 code (`shadow.reference_pseudobulk_totals`) on the confident calls, centred on the median informative gene and capped +-3, against the stored table on genes informative in both | warning when Pearson r < 0.9, recommending a PREP re-run with the in-house table as a new asset (not automatic) |
| Gene complexity | `gene_complexity_check` | median genes per cell of native vs simulated cells per depth bin (>= 50 cells each) | warning when native cells carry > 45% more genes; its text says simulated coverage predictions are unreliable for the dataset |

The thresholds are the `real_qc` fields of the annotation config
([Configuration](../configuration.md)).

**Outcomes and effects (M13).** Every check records one outcome per dataset
(`pass`, `warn`, `fail`, `not_applicable` or `not_evaluable`;
`RealQcProvenance`). A check that does not apply is `not_applicable`: paired
concordance on an unpaired section, the factor re-measure for a family
without an R3 member (no measured factor table) or on a dataset other than
the family's first, the prefilter spot check without a prefilter, the
version-7 checks on a version-6 bundle. Whether a check applies comes only
from facts the caller must state (`RealQcSignals`: the resolvability
version, whether the section is paired, whether the bundle is prefiltered,
whether the family has an R3 member). A check that applies but whose input
is missing is `not_evaluable`, so an input left out is never recorded as
`not_applicable`. A failing check can only lower the dataset: cap its gate
level (`gate_cap`), make an emitted level `not_resolvable` for it
(`withhold_level`), withhold the pair's supercluster-level cross-platform
statistics (`withhold_pair_stats`) or lower trust (`downgrade`).
`apply_qc_to_gate` never raises a gate level, `apply_qc_outcomes` never
raises a trust state, and `apply_qc_to_statuses` gives the statuses a
QC-applied RESOLVE produces: confident sets only shrink and keep their
labels. These three are property-tested over every outcome combination. The
combinators take no emission plan, floor plan or threshold, so they cannot
change one. The comparison against a QC-free re-run of RESOLVE
(pre-registration NR1) comes with the RESOLVE wiring (M13 chunk C15).
`real_data_qc` runs every check from the `real_qc` config:

| Check | Function | Rule | Effect |
|---|---|---|---|
| Marker referee (human) | `human_referee.human_marker_referee` (the statistic), `marker_consistency_outcome` | mouse G2 ported to the primary WHB bundle's profiles: per broad class, the panel genes that pass the §8.6 specificity rule (>= 20x and >= 1/1000; immediate-early genes excluded; classes with < 3 markers left out); table cells pseudo-labelled with `data/P1212`'s rule (>= 1.5 units and >= 60% of the marker units); the share of pseudo-labelled cells with a confident `ct_broad` whose call equals the pseudo-label | warning below 0.75; gate capped at `broad_only` below 0.70 (trust and margins unchanged); `not_evaluable` with fewer than 2 classes with markers or fewer than 200 scored cells |
| Registration G1 (human) | `registration_g1_outcome` | §7.6 as defined there: the fail rule is density ratio < 1.5 or shift > 5 µm; otherwise the warning rule is density ratio < 2.0 | fail rule: warning (`registration_g1_effect`; the gate fails with `gate_failed`); warning rule: warning |
| Paired concordance | `paired_concordance` | the pair's soft broad JSD on the shared-tissue mask (point estimate; the whole section reported) above 0.20 | supercluster-level cross-platform statistics withheld; `not_applicable` for an unpaired section; `not_evaluable` without a shared-mask value (the whole section is never scored in its place) |
| Flag rates | `flag_rate_summary` | more than half of the (class x platform) strata of the contamination, diffuse and spill-over flags uninformative under H16's 15% marking (a stratum is uninformative when any of its flags is) | warning; the per-flag, pooled and own-switch readings are reported beside it |
| Prefilter spot check | `prefilter_spotcheck` | agreement of the prefiltered with the unfiltered lookup below 0.95 at an emitted level | that level `not_resolvable` for the dataset |

**Human marker referee (M13 chunk C13).** `merxen.annotation.human_referee`
groups the WHB superclusters of the primary bundle's leaf level into the
seven broad classes (the WHB vocab; sinks and the nodes outside the seven
classes belong to none) and derives each class's markers on the panel's
query genes with the §8.6 / E3 specificity rule
(`flags.specific_gene_ratio`, `flags.specific_gene_min_share`).
`real_qc.marker_referee_comparator` chooses the comparison: `node` (the
default, mouse G2's rule: the class's unweighted mean supercluster profile
against every other supercluster, sinks included) or `class` (the class
profile, the `n_cells`-weighted mean of its superclusters'
`expected_fraction`, against the other classes' profiles). The `class`
profile is not `flags.class_profiles`, which weights `mean_cpm` and
renormalises over the query genes; the two bases give different sets. The
comparators differ where a class holds a small node that shares another
class's genes (the Committed oligodendrocyte precursor supercluster, in the
OPC class, shares oligodendrocyte genes) or a sink resembles a class
(Splatter): `node` then leaves that class without markers. The sets depend
only on the bundle and the panel. `RefereeMarkers.to_frame` writes them
before a run with their provenance (`source`, `comparator`, `min_ratio`,
`min_share` on every row), `RefereeMarkers.from_frame` reads them back
(`frozen`), and the run's QC details record their sha256 (`fingerprint`;
`frozen_fingerprint` for the table as read). A frozen marker that a dataset's
query genes lack is dropped and listed (`missing_genes`). Supplied derived
sets must record the run's comparator and specificity rule. Only `derived`
sets (D27 (a)) drive the outcome: hand-curated sets
(`RefereeMarkers.from_symbols`, `source` `hand_curated`, D27 (b)) and tables
without provenance (`supplied`) are scored by the same rule for reporting,
and `signal()` refuses them. Profiles without `n_cells` load; the `class`
comparator is then `not_evaluable`. The 200-cell minimum counts scored
cells (pseudo-labelled and confidently called), not pseudo-labelled cells
alone.
The 0.75 / 0.70 thresholds came from the H9 hand lists, so the derived
statistic is re-measured on set a before a new family is scored (M13 C17).
With the default `node` comparator set a's panel gives only one class with
three or more markers, so the referee is `not_evaluable` on set a: that is not
a pass, and it goes to the user with the threshold question.

On a seeded real-data family whose species gate has not merged into `main`
(`real_qc.seeded_families_warn_only_until_gate`), the lowering effects of the
referee, paired concordance, flag rates, gene complexity and the spot check
are demoted to warnings. Every other check applies as defined, registration
G1 included. The dataset gate's own outcome (and mouse G1 and G2) is recorded
from the gate verdict, never applied twice; without a verdict it is
`not_evaluable`.

## Pipeline processes

Four CPU processes in `workflows/modules/annotation.nf` and
`CLUSTERING_SQUIDPY_COMPUTE_CPU` in `workflows/modules/clustering_squidpy.nf`,
wired by `workflows/subworkflows/annotation_references.nf` (PANEL, PREP) and
`workflows/subworkflows/clustering_map_first.nf` (MAP in
`CLUSTERING_ANNOTATE_MAP`, RESOLVE after it in `CLUSTERING_ANNOTATE`,
COMPUTE_CPU after RESOLVE in `CLUSTERING_MAP_FIRST`); none takes the GPU
lock, and a run whose rows are all legacy instantiates none of them.

| Process | Runs | Resources | What it does |
|---|---|---|---|
| `ANNOTATE_PANEL` | once per pair × segmentation | 1 CPU, 4 GB | `merxen annotation-panel` on the gene list (`--annotation_panel_genes_path`) or on the pair's prepared H5ADs; writes the declared panels and `required_bundles.json`. |
| `ANNOTATE_REFERENCE_PREP` | once per unique (species, reference, `panel_hash`) across the run | 8 CPUs, 64 GB, 8 h; above 1,000 panel genes `annotation_prep_large_memory` and 24 h; one at a time on dwight | `merxen annotation-reference-prep`: gets the bundle from the store or builds it, and writes `bundle_ref.json`. Seconds when the bundle exists. |
| `CLUSTERING_SQUIDPY_ANNOTATE_MAP` | once per pair × segmentation, after its last required bundle (`map_first` only) | 6 CPUs, 24 GB (48 GB above 1,000 panel genes); `annotation_max_forks` (2) at a time on dwight | `merxen annotate` on the pair's prepared H5ADs with exactly the bundle refs PREP resolved: the MapMyCells runs, the tidy parquets, the provisional labels and `map_manifest.json`, published to `<outdir>/<pair>/<seg>/annotation_map/annotation_map_out/`. |
| `CLUSTERING_SQUIDPY_ANNOTATE_RESOLVE` | once per pair × segmentation, after its own MAP (`map_first` only) | 2 CPUs, 16 GB (32 GB above 1,000 panel genes); `annotation_resolve_max_forks` (4) at a time on dwight | `merxen annotate-resolve` on the MAP output, the prepared H5ADs, the panel and the same bundle refs: the label tables, annotation manifests and `<pair>_resolve_summary.json`, published to `<outdir>/<pair>/<seg>/annotation_resolve/annotation_resolve_out/` (see [Resolving](#resolving-merxen-annotate-resolve-m4)). 36-63 s per human pair × segmentation in the M4 shadow runs. |
| `CLUSTERING_SQUIDPY_COMPUTE_CPU` | once per pair × segmentation, after its own RESOLVE (`map_first` only) | 8 CPUs, 32 GB, main environment, no GPU lock; `clustering_squidpy_max_forks` (4) at a time on dwight | `python -m merxen.clustering_squidpy_stages compute --mode map_first` on the prepared H5ADs and RESOLVE's label tables: the map-first hierarchy, legacy-compatible `obs` columns, QC embedding and stability diagnostic, under the legacy file names (`<sid>_clustered.h5ad`), so the shared `CLUSTERING_SQUIDPY_FINALIZE` writes the clustered table. 15-28 min and 2.3-3.5 GB per human proseg_hybrid section in the M5 stage A runs. |

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
with `panel_status: refused` and its reasons, maps nothing, and RESOLVE
then writes statuses only. With `annotation_reuse_published` a run whose
query fingerprint, `build_hash`, engine parameters, ctm version, tidy schema
version and (restricted) lookup equal the published manifest's is copied
from `annotation_map/annotation_map_out/` instead of re-mapped, because
dwight prunes work directories; with `annotation_keep_extended_json` the
published run must have kept its JSON too. A published manifest that cannot
be read (the `-stub-run` manifest, an older or newer layout) only disables
reuse, with a warning. In a pipeline run MAP runs only in `map_first` mode
([Map-first clustering runs](#map-first-clustering-runs-m5)); the shadow
evaluation uses the standalone command.

RESOLVE starts for a pair × segmentation as soon as its own MAP has
finished. It stages the MAP output, the prepared H5ADs and clustering config
MAP read, the panel directory and MAP's bundle refs, and resolves every run
with exactly the bundle its staged ref names (`--require-bundle-refs`: a ref
whose `build_hash` differs from the one the run mapped with fails the task
as a stale MAP output; the store is never searched). The shared tissue mask
of the pair JSD comes only from ALIGN's output channel, which M5 wires (as
for `ANNOTATE_PANEL`); until then RESOLVE reports the whole-section JSD and
never looks for a published `align_out` that ALIGN may still be writing
(`--no-alignment-lookup`). A mouse MAP output is resolved with the M6
mouse rules ([Mouse rules v1](#mouse-rules-v1-consensusresolve_mouse-73-m6));
the task does not stage the QC stage's registration check yet (the QC
channel is pipeline wiring, M5), and a pipeline RESOLVE
(`--require-bundle-refs`) refuses a mouse sample without it, so mouse
`map_first` needs M5's wiring ([Mouse region step](#mouse-region-step-m6),
pipeline wiring). RESOLVE caches on content (`cache
"deep"`), as MAP does, and its task hash also sees the annotation config it
writes and a fingerprint of the RESOLVE rules: the sha256 of the files under
`src/merxen/annotation/`, `src/merxen/assets/annotation/` (floors,
vocabularies, validated panels, state genes), `src/merxen/clustering/` and
`src/merxen/cli/run_annotation.py` of the running checkout (`__pycache__`
skipped; `AnnotationReferences.RESOLVE_RULE_SOURCES`). So after a threshold,
floor, trust, flag or degraded-mode change (`annotation_allow_single_method`,
the only RESOLVE-only param; PANEL, PREP and MAP never see it), `-resume`
re-runs every RESOLVE, a minute each, and keeps MAP cached: MAP's task
inputs and its published-output reuse key hold none of these settings. Any
edit under those source directories re-runs RESOLVE, and so everything that
reads its tables.

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

## Map-first clustering runs (M5)

Human runs cluster in `map_first` mode by default since the M8 flip
(pre-registration §20); mouse stays `legacy` until M9. A run selects its
mode with `--clustering_squidpy_mode` (both species) or
`--clustering_squidpy_mode_human` / `_mouse`, and a samplesheet row's
`clustering_squidpy_mode` column (`legacy` / `map_first`) overrides the run
for that row (§20 D15), so one run can opt single datasets in or out. Human
`legacy` still works but is deprecated and logs a warning. After the flip a
human `map_first` row writes the unsuffixed clustered tables (the legacy
tables are replaced; keep a snapshot, or set
`--clustering_squidpy_table_key_suffix mapfirst`, if you need them), MENDER
uses the `exclude_from_features` policy for unassigned cells, the legacy
`mapmycells` stage runs only when `--annotation_mode_mapmycells_stage legacy`
and the row stops at `mapmycells`, and the run needs the reference-source and
reference-store params, which preflight checks. A header that only resembles
`clustering_squidpy_mode` (other case, spaces or hyphens) is ignored with a
warning, and those rows follow the run's mode. Hook
H5 in `workflows/main.nf` sends `map_first` rows through
`CLUSTERING_MAP_FIRST` instead of the legacy GPU `CLUSTERING_SQUIDPY_COMPUTE`
(a run whose rows are all `legacy` keeps the legacy wiring unchanged);
`CLUSTERING_SQUIDPY_PREPARE` and `CLUSTERING_SQUIDPY_FINALIZE` are the legacy
processes, unchanged:

```text
PREPARE -> ANNOTATE_PANEL -> ANNOTATE_REFERENCE_PREP (per unique bundle)
        -> CLUSTERING_SQUIDPY_ANNOTATE_MAP -> CLUSTERING_SQUIDPY_ANNOTATE_RESOLVE
        -> CLUSTERING_SQUIDPY_COMPUTE_CPU -> FINALIZE -> cortical depth, MENDER, ...
```

- **Preflight.** A `map_first` row gets no legacy marker, taxonomy or
  membership check; it is checked for the annotation settings, the
  references' source params (one full alternative of each required set),
  existing optional paths, the self-map sources while resolvability is on,
  and a writable reference store (and large-panel store when set). An empty
  `clustering_squidpy_table_key_suffix` is refused while the species has not
  flipped.
- **Samplesheet columns.** A row's own `anatomical_region` (human) or
  `mouse_section_regions` (mouse) is carried into each sample of its
  `samples_json` (and so into the clustering config); rows without the
  column inherit `annotation_human_region` / `annotation_mouse_section_regions`
  and keep the legacy samples JSON.
- **Alignment.** A paired, aligned pair's `shared_tissue_mask.npy` and
  `registration_summary.json` come from ALIGN's output channel (the ALIGN
  task of the run, or the published `align_out` when ALIGN does not run in
  it) and are staged into ANNOTATE_PANEL (label-free set c) and RESOLVE
  (the pair JSD's shared-mask restriction); a pair without alignment gets
  none.
- **COMPUTE_CPU** stages RESOLVE's label tables, annotation manifests and
  pair summary as files and caches on their content (`cache "deep"` hashes a
  file's content but a directory only by its metadata): a RESOLVE re-run
  with byte-identical outputs leaves it cached, a changed label re-runs it.
  Its task inputs also carry the run's table-key suffix, MENDER policy and a
  fingerprint of the hierarchy code (`AnnotationReferences.HIERARCHY_SOURCES`),
  so a hierarchy change re-runs it under `-resume`.
- **Clustered tables.** Before the species' flip, FINALIZE writes
  `<source table>_clustering_squidpy_mapfirst` (plan §4.8) and never the
  legacy key: the clustered H5AD records its suffix, and FINALIZE (whose
  legacy script names none) writes under it and refuses any other. Cortical
  depth reads that table (`clustered_table_key_suffix` in its table config)
  and MENDER receives its key through the row settings. The run publishes
  its clustering outputs to the same
  `<outdir>/<pair>/<segmentation>/clustering_squidpy/` as a legacy run, so
  snapshot legacy outputs before a `map_first` run into the same outdir
  (plan §2.3, M-1), or add its outputs beside them with a launch config
  ([Into a published results directory](#into-a-published-results-directory)).
- **MENDER.** The clustered table records the run's
  `mender_unassigned_state_policy` (default `exclude_from_features`), which
  MENDER applies to it (its legacy script names no policy); legacy tables
  keep `state`. MENDER_PREPARE and MENDER_IMPORT refuse a clustered H5AD
  whose mode or suffix does not match the table key they write (a
  `map_first` H5AD only to `<...>_clustering_squidpy_<its suffix>`, a legacy
  H5AD never to a suffixed key): the two tables hold the same cells, so the
  cell-id checks alone cannot tell them apart. A table whose states are all
  unassigned (a refused panel, a failed gate) is not an error: MENDER
  records `status: skipped_no_assigned_state`, writes its manifests without
  domains and imports nothing. The MENDER barrier of a run with any
  `map_first` row pairs every terminal event of a pair with the pair's one
  barrier spec (`combine`, also for that run's legacy rows; a run whose rows
  are all legacy keeps its `join`, which pairs items one to one, so a pair's
  earlier FINALIZE event would consume the spec, [MENDER](mender.md)).
  `map_first` runs never run the legacy MAPMYCELLS stage unless
  `annotation_mode_mapmycells_stage` is `legacy` and the run stops at
  `mapmycells`, so the barrier never waits for it.
- **Cross-platform scope.** COMPUTE_CPU records the pair's cross-platform
  scope in each clustered table
  (`uns["merxen_hierarchical_clustering"]["cross_platform_json"]` and
  `cross_platform_statistics_level`, and the hierarchical manifest), and
  MENDER and cortical-depth manifests carry it in their `annotation`
  record. The scope is RESOLVE's `pair.cross_platform`, built from the
  panels, with each sample's dataset gate folded in (plan §5.4): a
  `broad_only` gate caps the pair at `broad_only` (flagged, reason
  `dataset_gate:<sample>:broad_only`; P1212, whose MERSCOPE side is
  broad-only, is `broad_only` although its panel allows `full`), a `failed`
  gate sets `none`. A `broad_only` pair may be compared across platforms
  at the broad class and lineage only, a `none` pair not at all
  (`merxen.clustering.cross_platform.CrossPlatformScope.allows`), and only
  the composition kinds RESOLVE compared may be (`allows_kind`: a
  `per_platform` pair compares its `_xpanel` runs without the confident
  kind). Every cross-platform consumer applies this scope, not RESOLVE's
  record alone. MENDER niches are comparable across platforms only on
  `ct_mender_state` within such a scope (`cross_platform_comparable` in
  the MENDER manifest; [MENDER](mender.md)).
- **End-of-run summary.** A `map_first` run logs which pair × segmentation
  branches entered clustering, which got no label tables (a failed PANEL,
  PREP, MAP or RESOLVE task, dropped under `errorStrategy "ignore"`) or no
  hierarchy, the refused, broad-only and provisional panels, the broad-only
  and failed dataset gates, the pairs whose cross-platform statistics are
  restricted (by the panels or a dataset gate, the scope above) and the
  samples whose MENDER run was skipped for want of an assigned state. A
  legacy run prints nothing.

### Into a published results directory

A `map_first` run into an outdir that already holds legacy results would
change published files: FINALIZE, MENDER_FINALIZE / MENDER_IMPORT and
COMPUTE_CORTICAL_DEPTH publish to the legacy `clustering_squidpy/`,
`mender/` and `<platform>/compute_cortical_depth/` (all `overwrite: true`),
cortical depth rewrites the segmentation's base table
(`table_MOSAIK_proseg_hybrid`, not the clustered one) with its depth columns
while `cortical_depth_write_spatialdata_table` is `true`, PREP publishes to
`<outdir>/annotation_reference_prep/`, and Nextflow writes its report,
timeline and trace to `<outdir>/nextflow/`. The M5 exit run (P7513 and
P1212, proseg_hybrid, 2026-09-29) added its results beside the legacy ones
without changing any of them with a launch config (`-c`, which overrides
the process definitions and both profiles) and three flags. On 8 CPUs and
64 GB (the whole run) it took about 5 hours in two launches (COMPUTE_CPU
39-43 min per pair, MENDER_COMPUTE 11-26 min per sample; peak RSS 25 GB):

```groovy
process {
    // Every publishDir of the run gets overwrite: false; the colliding ones
    // move to new *_mapfirst directories. MENDER and cortical depth read
    // FINALIZE's task output, not its published copy.
    withName: "CLUSTERING_SQUIDPY_FINALIZE" {
        publishDir = [path: { "${params.outdir}/${pair_id}/${segmentation}/clustering_squidpy_mapfirst" }, mode: "copy", overwrite: false]
    }
    withName: "MENDER_FINALIZE" {
        publishDir = [path: { "${params.outdir}/${pair_id}/${segmentation}/mender_mapfirst" }, mode: "copy", overwrite: false, pattern: "mender_out/**"]
    }
    withName: "MENDER_IMPORT" {
        publishDir = [path: { "${params.outdir}/${pair_id}/${segmentation}/mender_mapfirst/mender_out/${platform.toLowerCase()}" }, mode: "copy", overwrite: false]
    }
    // Copy only the output directory: latest_input.zarr is a link to the store.
    // The pattern names the directory itself: a publishDir pattern is matched
    // against each declared output, so "compute_cortical_depth_out/**" matches
    // nothing and publishes an empty directory (the M5 exit run's first launch
    // did that; its second launch re-ran cortical depth with this pattern and
    // published the outputs).
    withName: "COMPUTE_CORTICAL_DEPTH" {
        publishDir = [path: { "${params.outdir}/${pair_id}/${platform.toLowerCase()}/compute_cortical_depth_mapfirst" }, mode: "copy", overwrite: false, pattern: "compute_cortical_depth_out"]
    }
    withName: "ANNOTATE_REFERENCE_PREP" {
        publishDir = [path: { "<somewhere outside the results root>/annotation_reference_prep/${bundle.reference_id}/${bundle.panel_tag}" }, mode: "copy", overwrite: false]
    }
    // ANNOTATE_PANEL, ANNOTATE_MAP, ANNOTATE_RESOLVE: their default paths,
    // with overwrite: false.
}
report.file = "<evidence dir>/report.html"      // likewise timeline, trace, dag
```

```bash
nextflow -c map_first_into_published.config run workflows/main.nf -profile dwight,conda \
    --samplesheet <rows with start_stage clustering_squidpy, stop_stage mender> \
    --outdir <published results> --clustering_squidpy_mode map_first \
    --clustering_squidpy_table_key_suffix mapfirst \
    --mender_enabled true --cortical_depth_write_spatialdata_table false
```

Since the human flip (pre-registration §20) a human `map_first` run writes
the unsuffixed clustered tables unless `--clustering_squidpy_table_key_suffix
mapfirst` is given, as above; without it, this recipe would replace the
published legacy tables.

Keep the rows' `stop_stage` at `mender` (or unset): a `map_first` row that
stops at `mapmycells` with `--annotation_mode_mapmycells_stage legacy` runs
the legacy MAPMYCELLS (hook H3), which publishes to the legacy `mapmycells/`
(a human row's default, `skip`, does not). `-preview` does not show which tasks a row
gets (it builds the static DAG of every invoked process without reading the
samplesheet); a `-stub-run` on a synthetic copy of the rows' layout does.
The stores then gain `tables/<table>_clustering_squidpy_mapfirst` and a
re-consolidated root `zarr.json`. Two properties of the consolidated
metadata: a table write re-consolidates from disk, so it also drops entries
of groups that are no longer on disk (P1212 Xenium carried 93 of a removed
`.table_MOSAIK_proseg_hybrid.merxen-backup-*` group); and
`write_or_replace_element` replaces a table (MENDER_IMPORT on the
`_mapfirst` table, and every legacy replacement) through such a backup
group and consolidates while it exists, so the root keeps stale entries of
the removed backup. SpatialData readers skip them.
`scripts/annotation/remove_mapfirst_outputs.py` removes a run's
`_mapfirst` tables, their (and their backups') consolidated entries and
its map_first output directories again, moving everything into a
quarantine directory on the same disk (dry run by default; `--apply`).
The unsuffixed `annotation_panel/`, `annotation_map/` and
`annotation_resolve/` do not record which run wrote them (a later MAP
reuses a published `annotation_map`), so it moves them only with
`--include-annotation-dirs --new-paths <the paths the run created>`; with
`--new-paths` anything not on the list stays.

With the default publishDirs (no launch config), a `map_first` run
publishes FINALIZE into `clustering_squidpy/` and MENDER into `mender/`,
the legacy directories: fine for a fresh outdir, but into an outdir with
legacy results it overwrites them, so use the launch config above. A
MENDER-only restart of a `map_first` row reads the clustered H5AD from
`clustering_squidpy_<suffix>/` when that holds it, else from
`clustering_squidpy/`, and MENDER refuses a legacy H5AD found there.

Three more things the M5 exit run showed:

- **`-resume` after a store write re-runs everything.** VALIDATE_ANALYSIS_LAYER
  and COMPUTE_CORTICAL_DEPTH take the store as a `path` input, which the
  standard cache hashes by the root directory's size and mtime; every table
  write re-consolidates the root `zarr.json` and changes that mtime. A
  `-resume` after FINALIZE has written therefore re-runs VALIDATE, whose new
  work-directory link enters the samples JSON of PREPARE and FINALIZE, so
  both run again (MAP, RESOLVE and COMPUTE_CPU stay cached only if their
  deep caches see byte-identical inputs), including FINALIZE's table write. The exit run's second launch (MENDER after the barrier fix)
  reset the four root directory mtimes to their pre-run values first, so
  only PREP (never cached), cortical depth (hashed after FINALIZE) and MENDER
  ran again.
- **MENDER needs the barrier fix.** With the legacy `join`, a `map_first`
  row with cortical depth enabled never ran MENDER: FINALIZE's event consumed
  the pair's one spec. `map_first` runs now `combine` every event with it
  ([MENDER](mender.md)).
- **Writer lock.** FINALIZE takes `<zarr>.merxen-write.lock` beside the path
  it is given, which is VALIDATE_ANALYSIS_LAYER's staged link in the work
  directory, so its lock file lands in the work directory and does not
  exclude writers that use the store's real path; MENDER_IMPORT resolves the
  real path and locks beside the store. Do not run other writers on the
  same stores while a run writes.

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
   `RESOLVABILITY_VERSION`, and of those the one whose human held-out
   self-map has the current test-set revision
   (`HO_SELF_MAP_TEST_SET_REVISION`, M8 D1; read from the bundle's hashed
   `test_set.self_map_exclusion`, else its `test_set_exclusion` record; a
   mouse bundle has no revision). When only older revisions exist it keeps
   them and logs a warning naming each bundle's revision. When none has the
   current version (a panel not rebuilt since the last bump), it takes the
   older bundle and logs a warning naming its resolvability version: its
   self-map tables are stale until the panel is rebuilt (a pipeline run's
   PREP rebuilds it).
3. Run MapMyCells (seed 0, bootstrap factor 0.5, 100 iterations, raw
   normalisation, one BLAS thread per worker, `--drop_level
   CCN20230722_SUPT` for WMB) and parse the extended JSON at once into the
   tidy parquet; the JSON is deleted unless `annotation_keep_extended_json`.
   Mouse maps unpruned first; the region step below then prunes.
4. Mouse: the region step ([Mouse region step](#mouse-region-step-m6)).
5. Write `map_manifest.json`: per sample the input identity, table-cell
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
| `<platform>/<sid>_mouse_regions.parquet` | Mouse: one row per table cell: `tile_i` / `tile_j` (150 µm tile), `tile_region` (the tile's majority home division), `inferred_region` (that division when present), `confident_neuron`, `region_dropped_level` (`class`, `subclass` or null) and `region_pruned_changed`. |
| `<platform>/<sid>_mmc_wmb_panel_pruned.parquet` | Mouse, when nodes were dropped: the tidy table of the re-mapped cells only (`--nodes_to_drop`). The unpruned run keeps `<sid>_mmc_wmb_panel.parquet`. |

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

### Mouse region step (M6)

`merxen.annotation.mouse_regions` (plan §7.2; the E7 design,
`exp/E7/09_autoregion_v2.py` and `05c_hybrid_droplist.py`) runs on each
mouse sample's unpruned WMB run:

1. **Request.** The sample's `mouse_section_regions` (the samplesheet
   column, read from the `samples` of `clustering_squidpy_config.json`;
   pipeline wiring: M5, see below), else the annotation config's
   (`annotation_mouse_section_regions`, default `auto`); `merxen annotate
   --mouse-section-regions` overrides both. `auto` infers the divisions, a
   `;`-separated list of CCF divisions (`Isocortex`, `OLF`, `OB`, `HPF`, `CTXsp`, `STR`, `PAL`, `TH`, `HY`, `MB`,
   `P`, `MY`, `CB`; case-insensitive) overrides the inference, and `none`
   disables pruning (no region-share bundle needed).
2. **Inference** (`auto`; also run for QC with an override): confident
   neurons (a neuronal class, 01–29, with class and subclass bootstrap
   probability ≥ 0.9) vote for their subclass's MERFISH home division
   (`wmb_region_share`: OB split from OLF); each 150 µm tile with ≥ 3 votes
   takes its majority home (ties: the first division in sorted order); a
   division is present with ≥ 1% of the assigned tiles and a connected
   (8-neighbour) component of ≥ 3 tiles. Fewer than 200 assigned tiles, or
   no `obsm["spatial"]`, skip pruning with a warning (status
   `skipped_few_tiles` / `skipped_no_coordinates`).
3. **Two-tier drop rule.** A class with less than 20% of its MERFISH
   grey-matter cells in the present divisions (and at least 100 such cells:
   E7's `n_grey >= 100`, so `15 HY Gnrh1 Glut`, 55 cells, never takes the
   strict rule; `min_merfish_cells_class`) drops its subclasses with a
   present share below 0.3, and its subclasses with fewer than 20 MERFISH
   cells follow it; any other class drops its subclasses below 0.1. A class
   losing every subclass is dropped as one node. Astro-Epen, OPC-Oligo,
   Vascular, Immune and Pineal, and classes without MERFISH grey cells, are
   never dropped. For the manual ag7 / VZG2 set (Isocortex, HPF, OLF, CTXsp,
   TH, HY, MB) this gives E7's 60 nodes exactly.
4. **Pruned re-map.** Only the table cells whose unpruned class or subclass
   was dropped are mapped again, with `--nodes_to_drop` and the bundle's
   lookup filtered to the pruned tree (keys of dropped nodes removed,
   markers restricted to the query, marker-less parents collapsed), so
   `cell_type_mapper` never meets a zero-marker parent. Every other cell
   keeps its label by construction, and also its unpruned bootstrap
   probability, which still counts the runner-up votes on dropped nodes:
   against a full pruned re-map with the same seed, 1.31% (ag7
   proseg_hybrid) and 0.69% (VZG2 original_seg) of the table cells are
   class-confident only in the full re-map and none only in the subset
   (subclass: 1.19% / 0.50% vs 1.05% / 0.54%, re-draw noise). The subset
   re-map is therefore conservative at class (a one-directional coverage
   deficit, never an extra confident call); plan §7.2 keeps it [L]. The
   re-map is reused from `--reuse-from` like a run (same fingerprint,
   bundle, engine parameters, ctm version, lookup and drop list).

The sample's `mouse_regions` record in `map_manifest.json` holds the
request, status, tile counts per division, the inferred and the used
divisions, the drop list (level, label, name, share, rule), the cells
dropped and changed, the re-map run and the region-share bundle's
`build_hash`. The provisional labels use the pruned calls (`mmc_wmb_*`) and
add `mmc_wmb_unpruned_{class,subclass}_{name,bp}`, `region_pruned_changed`
and `inferred_region`. `load_resolve_runs` gives RESOLVE the pruned table as
the primary run's `tidy` (the unpruned one as `unpruned_tidy`) and the
region outputs (`ResolveRun.regions`) with the region-share bundle the step
used: `--bundle wmb_region_share=DIR` or its staged `--bundle-ref` when
given, else the recorded path; a bundle whose `build_hash` is not the one
MAP recorded is refused, and one that cannot be read for a pruned section
is a gate warning (G3 and G4 not evaluated). Mouse RESOLVE refuses a MAP
output without the region step (a manifest from before M6, whose
`mouse_region_step` is not the M6 step, or a mapped sample without a
`mouse_regions` record): re-run `merxen annotate`. A skipped step
(`skipped_few_tiles`, `skipped_no_coordinates`) and an override that
differs from the inferred divisions are gate warnings (plan §7.2), `none`
is not. The label table's provenance (`mouse_gate`) records the region
step's status and reasons, the drop list's size and sha256
(`drop_list_sha256`: sha256 of the JSON `[level, label]` pairs) and the
region-share bundle as reference `wmb_region_share`; the resolve summary
has the same under `region_step`, and under `region_coherence.per_class`
plan §7.2 step 5's class coherence against E7's intrinsic whole-brain
coherence (median `region_coherence` of the cells called to each class,
the class's MERFISH `coh_home_median` from
`wmb_region_restricted_classes.csv`, and their ratio; MERFISH is denser
than most query sections, so the ratio is a relative QC reading). Region
coupling (`coupled_regions`) and other rule variants are M6b and are
refused, not ignored. Sagittal, OB- and CB-dominated sections are out of
scope (use an override or `none`).

**Pipeline wiring (M5).** `CLUSTERING_MAP_FIRST` is not wired yet; when M5
wires mouse `map_first`, the tasks need: (1) each entry of the MAP task's
`samples_json` (`clustering_squidpy_config.json` `samples`) to carry the
row's value under `mouse_section_regions`, the
`ClusteringSquidpySampleConfig` field that
`samplesheet_columns.section_regions_by_sample` reads (the row settings'
`annotation_mouse_section_regions` key is not read by MAP); (2) the
RESOLVE task to stage the QC stage's registration check of each sample and
pass it with `--registration-qc` or `--registration-qc-dir` (a pipeline
RESOLVE, `--require-bundle-refs`, refuses a mouse sample without it); (3)
the `wmb_region_share` bundle ref staged for RESOLVE as for MAP; (4) a MAP
rules fingerprint (as RESOLVE's `rules_fingerprint`) over
`mouse_regions.py`, `config.py` and the MAP pipeline code, or
`MOUSE_REGION_STEP`, `REGION_CELLS_SCHEMA_VERSION` and
`config.mouse_regions` in the MAP annotation config, so a region rule
change re-runs the deep-cached MAP under `-resume`.

A published clustered H5AD of a small sample can have a `min_cells`-filtered
`var` (P1212 and P5011 reseg MERSCOPE hold 299 and 268 of 300 features).
Its declared panel is the control-filter record, so it keeps the prepared
panel's hash and bundles; the filtered-out features have no native ID there
and resolve by symbol (pair lookup, local gene table). MAP maps the genes
present and records the missing ones (a subset bundle when they matter).
A clustered H5AD without the record (written before M1) declares its `var`;
give it the declared panel with `merxen annotation-panel --panel-file
<PLATFORM>=<declared panel file>` and `--panel-dir`.

## Resolving (`merxen annotate-resolve`, M4)

The RESOLVE step (`merxen.annotation.pipeline.annotate_resolve`; plan §3.4)
turns a MAP output into one label table per sample (§4.1). The standalone
command runs it on published MAP outputs (options in
[CLI](../cli.md#merxen-annotate-resolve)); the `CLUSTERING_SQUIDPY_ANNOTATE_RESOLVE`
process runs it after each MAP in `CLUSTERING_ANNOTATE`, which
`CLUSTERING_MAP_FIRST` calls from M5 (legacy runs never do; see
[Pipeline processes](#pipeline-processes)).
Per sample:

1. **Inputs.** The counts are reloaded as MAP loaded them and must have the
   sample fingerprint MAP recorded; every tidy parquet must have its
   recorded sha256. A run is resolved with the bundle it mapped with, or
   with an override / the store's current bundle of the same reference and
   panel whose marker lookup is the one the run mapped with. The panel's
   family is re-derived from the current `validated_panels.csv`, and each
   reference's trust state is decided as PREP decides it
   (`diagnostics.panel_diagnostics` + `trust_for_panel`).
2. **Calls.** WHB: the supercluster call with its bootstrap probability,
   `avg_correlation`, best runner-up and margin; lineage, broad and NT sum
   the supercluster bootstrap probabilities over the assigned node's class
   (the bundle's vocab snapshot); SEA-AD: E2's 7-class label and broad
   probability and the subclass `aggregate_probability`.
3. **Emission.** The primary bundle's resolvability tables are re-decided
   with the simulated cells reweighted to the dataset's soft composition
   per depth bin (the assigned supercluster's bootstrap probability plus
   its runner-ups', on node labels; §8.3), giving per (level, class, depth
   bin) whether a level is emitted and, outside real-data-validated
   families, the local threshold (raise-only). Then the floors, the dataset
   gate (level + warning flag; a provisional panel warns and shows a banner
   but never lowers the level) and the degraded-mode consensus
   (`consensus.resolve_human`, §5.2–§5.4). A resolvability version-7 bundle
   (every family outside `validated_panels.csv` and the version-6 pins,
   such as the M13 custom MERSCOPE panel) is read the same way with its own
   rule: each ensemble member's cells are reweighted separately and the
   version-7 ensemble re-decided (E1, E2, the saturated-bp rule), then the
   monotone fill; cells of a filled bin are `resolvability_extrapolated`
   and floors come from the decisions before the fill (see "Resolvability
   version 7 in RESOLVE" below). Version 6 is read exactly as before.
4. **Flags** (`merxen.annotation.flags`, §4.3, §5.6): contamination on the
   assigned class's negative genes (negative in both WHB frontal and SEA-AD
   Multiregion, minus the state genes) against a beta-binomial fitted to
   the class's deep confident cells in this dataset (p < 0.01 with >= 3
   negative counts); the diffuse profile against the q95 of 200
   multinomial draws from the class profile; the robust z of
   `avg_correlation` within class × platform × depth bin (< -3). Realised
   rates are recorded per class × platform over the confident broad calls;
   a contamination stratum above 15% or a diffuse stratum above 30% is
   uninformative and its flag is null (H16's 15% marking is reported beside
   the diffuse rate). The microglial spill-over flag is off for human
   (OD-C5). `discovery_caution` is any of contaminated, diffuse, OOD or
   method disagreement. `flag_nonneuronal_high_depth` (report-only, version
   7 only; null otherwise) is described below.
5. **Composition** (`merxen.annotation.composition`, §5.5): the soft broad
   vector of every table cell (`soft_broad_*`, summing to 1 with
   `soft_broad_unallocated`), the section's soft, depth-stratified (>= 30
   counts), confident-only and argmax compositions, and for the pair the
   base-2 JSD of the renormalised seven-class vectors with a 95% spatial
   block-bootstrap CI (500 µm tiles, 200 replicates, joint resampling on
   the shared frame), whole section and inside the shared tissue mask. A
   `per_platform` pair's JSD and pair compositions come from the WHB runs
   on the intersection panel (`_xpanel`, §8.5; soft, soft ≥ 30 and argmax:
   the own-panel confident labels rest on different gene sets), recorded as
   `cross_platform.jsd_run`; with fewer than 100 intersection genes or a
   broad-only intersection (by resolvability) the cross-platform statistics
   are `broad_only`, flagged with reasons, for M5 / M7. The per-sample
   compositions and `soft_broad_*` stay on the own panel and are never
   compared across platforms.
6. **Outputs.** `<platform>/<sid>_celltype_labels.parquet` (validated;
   provenance in the parquet schema), `<platform>/<sid>_annotation_manifest.json`
   (`AnnotationProvenance`, the JSON `uns["merxen_annotation_json"]` holds)
   and `<pair>_resolve_summary.json`, all deterministic for given inputs
   (the label parquets hold no clock); the run record
   (`annotation_resolve_run.json`: clock, wall time, absolute paths) is
   published beside `annotation_resolve_out`, so COMPUTE_CPU, which stages
   the label tables, manifests and pair summary as files and caches on their
   content, re-runs only when a label changes (M5). A
   run that mapped with the parent bundle on a restricted lookup (a sample
   lacking panel genes; §3.3) is resolved with the parent's resolvability
   and trust, recorded as `resolvability_inherited` (`restricted_lookup`)
   with a trust reason. A refused panel gives statuses only
   (`not_attempted_gate`, gate `failed`, `exclude_hard`). Mouse samples
   follow the same steps with the mouse rules below.

**Resolvability version 7 in RESOLVE** (the M3c follow-up, plan §12 M3c;
M13 chunk C14). Human and mouse RESOLVE declare version-7 support
(`load_resolvability(..., allow_version_7=True)`); a resolvability version
this code does not know (later than 7, below 1, or not an integer) is refused
with a `ResolvabilityError` for every caller
(`resolvability.checked_resolvability_version`). For a version-7 bundle:

- **Decisions.** `ResolvabilityTables.decisions` re-derives the ensemble
  decisions from `resolvability_cells.parquet`: each emission member's cells
  are reweighted to the dataset's soft composition per depth bin (pooled deep
  sets per member, trimmed per tested set as in version 4), the members are
  decided with the saturated-bp rule, the ensemble rule (E1, E2 with the
  spread margin of the bundle's ensemble settings) judges each bin, then the
  monotone fill runs (plan §8.3 v7.7-v7.9). Without reweighting
  (`resolvability.reweight_to_composition = false`, or no leaf calls) these
  are PREP's unweighted decisions. Thresholds, targets and every rule
  constant are the bundle's; RESOLVE changes none of them.
- **Emission.** A filled bin is emitted at its threshold and its confident
  cells are `resolvability_extrapolated` (the decisions' `extrapolated`).
  Floors come from the decisions before the fill (`EmissionPlan.simulated_floors`;
  the fill only reaches bins deeper than the shallowest emitted one, so it
  cannot move a floor). The trust state is the bundle's (PREP's guarded
  constraint); RESOLVE never raises it.
- **Class-depth table.** `EmissionPlan.class_depth` is
  `resolvability.class_depth_table` of the decisions RESOLVE applied: the
  schema of PREP's `resolvability_class_depth.parquet`, but reweighted to the
  dataset. RESOLVE replaces the profile shares by the dataset's own per-class
  depth s_c(d): per level, the table cells whose class key there is c,
  binned by total counts; a class with fewer than 100 such cells
  (`real_qc.CLASS_DEPTH_MIN_CLASS_CELLS`, the per-class minimum of a depth
  profile, plan §8.3 v7.5; not pre-registered and awaiting confirmation,
  pre-registration §22.9) takes the label-free total-count histogram of all
  table cells (`share_source` `label_free`). The coverage warning judges only
  classes with at least `real_qc.coverage_min_cells` (200) cells, so at that
  setting it never uses the label-free histogram. The class-depth prediction
  sum_d s_c(d) cov(L, c, d) and the resolvable share sum_d s_c(d) over
  emitted bins (`real_qc.dataset_class_depth_prediction`) are recorded per
  (level, class) in the dataset's regime; they are the predictor of the
  per-class real-vs-simulated coverage check, which M13 wires (chunk C15).
- **`flag_nonneuronal_high_depth`** (report-only; plan §8.3 v7.9): a table
  cell is flagged when, at some level of the bundle, its class key there is
  non-neuronal (the bundle's `neuronal_classes`; unknown lineage counts as
  non-neuronal), its total counts reach `real_qc.nonneuronal_high_depth_counts`
  (1,000) and its (level, class, bin) is emitted on its own ensemble verdict
  and marked `nonneuronal_high_depth` (never filled). It changes no status,
  label, threshold, floor or trust state and does not enter
  `discovery_caution`. It is null outside the table, and for every cell of
  a version-6 bundle or a run without resolvability tables. The mark is set
  by the bin's grid depth, not the cell's: a bundle whose grid ends below the
  limit has no marked bins, and the flag is false for every table cell. The
  new-panel human MERSCOPE family's WHB bundle is one (its grid ends at 250
  counts), so glia at >= 1,000 counts sit in its open-ended 250 bin, which
  the fill's non-neuronal limit does not reach either (an open question,
  pre-registration §22.9).
- **Summary.** Each sample's entry in `<pair>_resolve_summary.json` gains
  `resolvability_v7`: the version, the decision recipe (`ensemble`), the
  emission members, the applied decisions' bins per regime (emitted,
  unanimous, by the spread, filled, at the saturated cap, non-neuronal high
  depth), the class-depth prediction per (level, class) (`n_cells`,
  `share_source`, `predicted_coverage`, `resolvable_share`) and the flag's
  count per level. A version-6 sample has no such entry.

Version-6 bundles (set a, set c, SEA-AD, ag7, VZG2, the pinned P5011
MERSCOPE family) and runs without tables give identical RESOLVE outputs
apart from the new, all-null label column (`test_resolve_v7.py`: golden
digests of the label tables, provenance and summaries computed with the code
before this change, floats to 10 significant digits, as for the version-6
resolvability tables of pre-registration §22.8; on the development host they
also matched at full precision). Label tables written before the column
still validate: it is an optional contract column
(`schema.OPTIONAL_CONTRACT_COLUMNS`), checked when present.

### Human rules v1 (`consensus.resolve_human`, §5.2)

Raw bootstrap probabilities (float32-tolerant: a stored 0.69 meets 0.69) are
compared with the v1 thresholds, raised where a bundle or, outside
real-data-validated families, a local resolvability threshold asks for more.

| Level | Confident when |
|---|---|
| Lineage | WHB lineage-summed probability >= 0.73, the node is plausible (not a sink, plausible for the region), counts >= the floor, resolvability emits it, and the second vote passes: below 60 counts SEA-AD agrees at lineage; from 60 SEA-AD does not confidently (>= 0.68) call another lineage. An implausible node keeps its lineage when SEA-AD agrees at lineage. |
| Broad | Confident lineage, broad-summed probability >= 0.73, the class × platform broad floor, resolvability, and the same vote at the seven-class level. **COP rule:** a WHB "Committed oligodendrocyte precursor" call is broad OPC only with >= 120 counts and supercluster probability >= 0.69, or when SEA-AD confidently calls OPC; otherwise it stays at lineage (`Oligodendrocyte lineage/unresolved` branch). `flag_cop_suppressed` marks every COP call on a confident lineage whose COP rule fails, whichever broad check decides its status (§4.3). |
| NT | Neurons only (others `not_applicable`): confident broad, the Exc / Inh probability, the broad floor, resolvability; no second vote. |
| Supercluster | Confident parent (NT for neurons, else broad), gate `full` (a warning does not block), probability >= 0.69, the supercluster floor (COP 120), resolvability. Requiring a confident NT for neurons is a recorded M4 deviation from plan §5.2 rule 4 (broad only; pre-registration §15): the final label is the deepest level of a contiguous confident chain. |
| SEA-AD subclass | Secondary name, never in `ct_final`: gate `full`, confident WHB broad that SEA-AD's class agrees with, the supercluster floor of the class, SEA-AD's threshold (0.55 below 60 counts, 0.45 from 60), the leaf's resolvability. |

The final label is the deepest confident level along lineage → broad → NT →
supercluster (`Mixed/Unknown` when none). When several checks fail the
status is the first of `low_counts` > `not_attempted_gate` >
`not_applicable` > `implausible` > `parent_unresolved` > `below_floor` >
`not_resolvable` > `low_confidence` > (COP rule) > `single_method` /
`method_disagree` ([statuses](../outputs.md#label-table-schema-sid_celltype_labelsparquet));
the SEA-AD subclass follows the same order, its SEA-AD agreement being its
`method_disagree`. `ct_consensus_tier` counts the agreeing informative
methods: 0 is a confident disagreement only, -1 no informative method.

**Degraded modes** (§5.3; `consensus.DEGRADED_MODES`, one truth table):

| Mode | When | Below 60 counts | Maximum tier |
|---|---|---|---|
| `whb_sea` | v1 default | SEA-AD must agree; from 60 it may veto | 2 |
| `whb_only` | SEA-AD failed, disabled or refused for the panel | `single_method` (not confident), unless `annotation_allow_single_method` (WHB decides alone, recorded) | 1 |
| `primary_missing` | WHB failed or refused | statuses only (`not_attempted_gate`) | 0 |
| `whb_sea_ll`, `whb_ll` | table rows only: the likelihood-typer vote is not enabled in v1 or v1.1 (OD-B8, decided 2026-09-27) | – | – |

**Dataset gate** (§5.4): `broad_only` when fewer than 30% of table cells
reach 30 counts (A < 0.30), `failed` when confident broad coverage of table
cells is below 0.25, otherwise `full`; a warning flag (never a lower level)
when confident broad coverage of segmented objects is below 0.15, or when a
simulation-validated family has more than 10% of its confident calls
outside the validated region. A provisional panel warns and shows the
banner. **Flags are report-only**: `discovery_caution` marks cells for
downstream sensitivity analyses (OD-B5: keep, covariate, with / without),
and no flag changes a status. The realised rates per flag × class ×
platform are in the resolve summary (`flags.strata`: `rate` over the
stratum's confident broad calls, `rate_all` over its table cells,
`informative` for the §4.3 switch, `informative_h16` for H16's 15% mark).

### Mouse rules v1 (`consensus.resolve_mouse`, §7.3; M6)

`merxen.annotation.mouse_resolve.resolve_mouse_sample` resolves a mouse
sample on the pruned view of its WMB run (re-mapped cells merged in, see
[Mouse region step](#mouse-region-step-m6)). One method, no second vote;
`ct_consensus_tier` is 1 when the class call meets 0.90, else -1.

| Level | Confident when |
|---|---|
| Broad | The vocab broad class of the call (Neurons, Astrocytes/Ependymal, Oligodendrocyte lineage, OEC, Vascular cells, Microglia); the class bootstrap probabilities of the call and its same-broad runner-ups sum to >= 0.90, counts >= the floor, resolvability. |
| Class | Confident broad, class bootstrap probability >= 0.90, the class floor (20 counts on ag7 / VZG2), resolvability, and `avg_correlation` >= `wmb_class_min_corr` when a floor is configured (real-data-validated families only; see the shadow study below). A cell whose unpruned class was region-dropped and whose re-mapped class is not confident is `implausible` at class and below (it falls back to broad). |
| NT | Neurons only: confident class, the NT-summed probability (Glut / GABA / Other from the vocab), resolvability. |
| Subclass (leaf) | Gate `full`, confident parent (NT for neurons, else class), bootstrap probability >= 0.80, the subclass floor (50), resolvability. A re-mapped cell (class or subclass dropped) whose new subclass is not confident is `implausible`. |
| Supertype | Report-only (`allow_fine_levels`, OD-E4): confident subclass, 0.80, resolvability; never a leaf or in `ct_final`. |

A call to a node the bundle lists in `marker_unsupported_nodes` (below an
auto-collapsed parent, or without a 10Xv3 marker cell, e.g.
`157 RN Spp1 Glut`) is `not_resolvable` at its level (the resolve summary
counts them).

**Class `avg_correlation` floor** (§7.3; pre-registration §16): the M6
shadow study scored floors 0.40 and 0.50 on ag7 proseg_hybrid and VZG2
original_seg against class coverage and MO1's marker consistency by depth
and selected **none** (0.40 changes consistency by < 0.001; 0.50 costs 7.7
/ 10.2 points of class coverage, and on VZG2 removes calls nearly as
marker-consistent as the kept ones), so `wmb_class_min_corr` stays unset
and `ct_class_corr` is report-only. Were a floor configured, it would apply
to the real-data-validated mouse families only (`avg_correlation` depends
on the number of markers; other panels need a simulation-derived floor). `ct_branch` and `ct_mender_state` are the confident class
(else `Mixed/Unknown`), `ct_leaf` the confident subclass (else
`unresolved`); `soft_class_<class>` holds the class bootstrap mass of the
call and its runner-ups over the 34 WMB classes (the rest is unallocated).

**Mouse gate** (`merxen.annotation.mouse_gate`, §7.6; label-free, computed
before the statuses; level + warning flag as for human):

| Signal | Source | Failed if | Warning if |
|---|---|---|---|
| G1 registration | `--registration-qc SID=PATH` or `--registration-qc-dir DIR`: the QC stage's `*_registration_qc.json` or `*_qc_summary.csv` (M0a check); required in a pipeline task unless `--no-registration-qc` | density ratio < 1.5, or shift > 5 µm against the platform's own cells | ratio < 2.0; no check given (standalone), or the check skipped |
| G2 marker referee | the table cells' class calls against marker pseudo-labels: six class groups (the vocab broad classes), each with the panel genes whose group profile is >= 20x every other class's profile (E3 rule; groups with < 3 genes left out), labelled with `data/P1212`'s rule (>= 1.5 units and >= 60% of the marker units) | consistency < 0.70 | < 0.80, or fewer than 200 pseudo-labelled cells |
| G3 implausibility | pre-pruning T2 share, E7's definition: unpruned subclasses with >= 20 MERFISH grey-matter cells, outside the never-drop classes, with < 25% of those cells in the present divisions; also MO2's "T2 after pruning" on the pruned calls. Not evaluated (a note) when the step was disabled; a skipped step warns (above) | – | > 3% |
| G4 composition | soft class shares against the pooled MERFISH sections of `--mouse-g4-sections` (the region-share bundle's `section_composition.parquet`; none by default until M6b's AP estimate) | – | Astro-Epen outside ±5 points or Immune outside ±1 point |
| G5 spill-over | `flag_microglial_spillover` rate | – | > 15% |

The primary panel's trust caps the level as for human (`refused` fails,
`broad_only` blocks the subclass) and a provisional panel warns. A `failed`
gate makes every table cell `not_attempted_gate` and `exclude_hard` (a
failed registration check: plan §7.5). G2's marker sets are derived per
panel, so the 0.70 / 0.80 thresholds, set on MO1's hand-listed referee
(.911 / .856), are provisional there [L]: the derived referee reads .805 on
ag7 proseg_hybrid and .819 on VZG2 original_seg (and .666 on the invalid
VZG2 proseg_hybrid).

**Mouse flags** (`merxen.annotation.mouse_flags`, §7.4, §8.6; report-only):

- `microglia_stat`, `microglia_weight`, `flag_microglial_spillover`: E3's
  specific-gene binomial likelihood-ratio test. The genes are derived from
  the bundle's profiles (microglia subclass vs every non-Immune class:
  ratio >= 20, share >= 1/1000, immediate-early genes excluded; ag7: Csf1r,
  Cx3cr1, Aif1, Blnk, C1qa, as E3; VZG2: E3's 13); flagged at statistic >= 10
  and spill weight >= 0.05. Null with fewer than 3 genes
  (`insufficient_panel_genes`) or when more than 0.5% of the marker
  astrocytes are flagged. Marker astrocytes follow E3's AST rule: >= 2
  counts and >= 10 per 1,000 on the panel's derived Astro-Epen genes, and
  at most 3 per 1,000 on E3's MG_loose purity genes (Cx3cr1, Csf1r, C1qa)
  when all three are on the panel, else the three most specific derived
  microglia genes. The purity filter bounds a marker astrocyte's counts on
  the purity genes, so the check fires through the flag genes outside that
  set (with purity on every flag gene the rate would be 0 by construction,
  and the check is then recorded as not evaluated). With >= 6 genes the flag is rebuilt on half of them and the
  held-out half's enrichment is reported (VZG2: 38x). It measures
  spill-over prevalence, never a microglia count. Per cell it agrees with
  E3 on 99.8% of ag7 and VZG2 cells (E3 used supertype profiles; the bundle
  has subclass profiles).
- `flag_region_incoherent`, `region_coherence` (F1, E7 §3): a cell of one of
  E7's 23 region-restricted classes (`wmb_region_restricted_classes.csv`)
  with kNN30 same-class coherence < 0.1 and class bootstrap < 0.8; null
  without coordinates.
- `flag_astro_lowcount`: an Astro-Epen call below 100 counts.
- Contamination, diffuse profile and OOD as for human, over the six mouse
  broad classes.

### Mouse exit on real data (M6)

The M6 exit (plan §12 M6; `$A/m6/stageC/M6_EXIT_REPORT.txt` of the
evidence archive) ran the standalone `merxen annotate` and `merxen
annotate-resolve` with the defaults above on the published clustered H5ADs
of ag7 (Ageing07) and VZG2 (the 2026-09-27 re-run with the offset fix),
every segmentation, and on the old VZG2 proseg_hybrid from before the fix
(misregistered by about 114 µm) as the gate's negative control. G1 used the
QC stage's registration check (VZG2 re-run) or the same M0a check computed
on the run's own transcripts (ag7, the old run); G4 the pooled
MERFISH-638850 sections .31–.33 (AP 8.0–8.4 mm).

**E7 reproduced.** On every section the inferred divisions are the manual
set (Isocortex, OLF, HPF, CTXsp, TH, HY, MB) and the drop list is E7's 60
nodes. The unpruned labels of ag7 proseg_hybrid and VZG2 original_seg are
identical to E7's inputs, and E7's full-density T2 counts (395 and 712
cells: subclasses with >= 20 MERFISH grey-matter cells, outside the
never-drop classes, with < 25% of them in the present divisions; E7's
definition, which G3 and MO2 use) fall to 4 and 0 after pruning (0.381% →
0.004% and 0.635% → 0; MO2 ≤ 0.1%). 394 and 712 cells are re-mapped
(39–43 s); every other cell keeps its label by construction, and a full
pruned re-map of every cell changes 0.245% / 0.060% of class and 1.24% /
0.42% of subclass labels outside the dropped nodes (MO2 ≤ 0.25% / 1.5%).
The cells the subset re-map leaves keep their unpruned bootstrap
probability: at class the full re-map only raises it (15.2% / 4.0% of
those cells up, 0.01% / 0 down; the dropped nodes' votes move to the kept
classes), so 1.31% / 0.69% of the table cells are class-confident only in
the full re-map and none only in the subset, a conservative coverage
deficit of the plan's subset re-map; at subclass the differences are
symmetric (1.19% / 1.05% and 0.50% / 0.54% confident only in one),
re-draw noise (`$A/m6/review/SUBSET_VS_FULL_CONFIDENCE.txt`).
The moved cells go where E7's did (ag7: MY Glut → MB Glut 122, → MB GABA
54, MY GABA → MB GABA 51; VZG2: MY Glut → MB Glut 475, → MB GABA 102).

| Section, segmentation | Table cells | Gate | G1 ratio / shift (µm) | G2 | G3 | G4 Astro-Epen / Immune (points) | G5 spill-over | Confident class / subclass | F1 | Astro low-count |
|---|---|---|---|---|---|---|---|---|---|---|
| ag7 proseg_hybrid | 103,736 | `full` | 2.20 / 0 | .805 | .0038 | +1.8 / −0.5 | .062 | .732 / .618 | 0.82% | 2.2% |
| ag7 reseg | 101,750 | `full` | 2.17 / 0 | .860 | .0037 | −3.6 / +0.3 | .056 | .823 / .706 | 0.81% | 1.6% |
| ag7 original_seg | 136,033 | `full` + warning (G1, G2, G4) | 1.63 / – | .773 | .0065 | +7.1 / −0.6 | .057 | .610 / .489 | 1.25% | 11.4% |
| ag7 proseg_mask | 103,766 | `full` + warning (G1) | 1.98 / 0 | .819 | .0040 | +0.8 / −0.3 | .059 | .727 / .613 | 0.79% | 3.3% |
| VZG2 original_seg | 112,158 | `full` | 2.13 / – | .819 | .0063 | +2.2 / +0.1 | .095 | .825 / .759 | 0.28% | 0.5% |
| VZG2 proseg_hybrid | 104,464 | `full` | 2.40 / 0 | .843 | .0060 | +1.3 / +0.4 | .096 | .859 / .794 | 0.24% | 0.3% |
| VZG2 reseg | 104,276 | `full` + warning (G4) | 2.43 / 0 | .903 | .0069 | −3.2 / +1.1 | .090 | .911 / .841 | 0.29% | 0.2% |
| VZG2 proseg_mask | 104,507 | `full` | 2.37 / 0 | .889 | .0057 | −1.0 / +0.8 | – (null: FPR) | .878 / .820 | 0.22% | 0.4% |
| VZG2 proseg_hybrid, old (invalid) | 102,134 | `failed` + warning (G1, G2; G4) | 1.34 / 114 | .666 | .0058 | +16.4 / −0.7 | – (null: FPR) | 0 / 0 | 0.58% | 3.0% |

Shares are of table cells; G3 is the pre-pruning T2 share in E7's
definition (the review re-run; equal to E7's counts on every section); "–"
under G1 is the platform's own segmentation (no shift test), under G5 a
spill-over flag nulled by its false-positive check. MO1 (marker consistency over
all marker-pseudo-confident cells) is .9118 on ag7 proseg_hybrid and .8558
on VZG2 original_seg (≥ .89 / .84); MO4 (soft class composition vs the
MERFISH window) holds on both. The soft class JSD against VZG2 original_seg
(MO9, scored at M9 [L]) is 0.027 for the re-run proseg_hybrid, 0.061 for
proseg_mask and 0.088 for reseg (base-2 distance over the 34 classes).

**Negative control.** The old VZG2 proseg_hybrid is `failed` (G1: density
ratio 1.34, shift 114 µm; G2: .666 < .70) with the G4 warning (Astro-Epen
+16.4 points), so every cell is `not_attempted_gate` and `exclude_hard`;
without a registration check G2 alone still fails it. The re-run
proseg_hybrid of the same section is `full` without a warning (G1 2.40, G2
.843). The spill-over rule's rate and the soft Astro-Epen share are the
signals that move on the invalid data (12.8% of the cells, and 0.57% of
its marker astrocytes, which nulls the flag; 33.8%); per-cell
confidence does not (class bootstrap ≥ 0.9 on .733 of its cells, .739 on ag7
proseg_hybrid).

**Flags** (report-only; rates over table cells): the spill-over flag fires
on 5.6–6.2% of ag7 and 8.9–9.6% of VZG2 cells (proseg_mask's 8.3% is
nulled, below; E3: 6.1% / 9.4%; spill-over prevalence, not microglia),
with a held-out enrichment of 38× on VZG2
original_seg (45–129× on the other VZG2 segmentations; ag7 has 5 derived
genes, no held-out test). The false-positive check (review re-run; E3's
MG_loose purity, the flag genes outside it: 2 on ag7, 10 on VZG2) flags
1–2 of 16,495–26,156 ag7 marker astrocytes (≤ 0.012%) and 0.37–0.46% of
VZG2's (original_seg 126 of 34,007: MO5 ≤ 0.5%), and nulls the flag on VZG2
proseg_mask (0.54%) and the old invalid run (0.57%). F1 flags
0.8% of ag7 (1.25% of original_seg) and 0.22–0.29% of VZG2 cells (MO6:
0.2–1.0% on ag7 proseg_hybrid and VZG2 original_seg). `flag_astro_lowcount`
flags 1.6–3.3% of ag7 cells (11.4% of original_seg, where 20% of the table
cells have fewer than 100 counts) and 0.2–0.5% of VZG2 cells.

**Uncovered subclasses.** The unpruned runs call the five uncovered
subclasses for at most 2 cells per section (none on ag7 proseg_hybrid or
VZG2 original_seg), and the pruned re-map sends 0–2 cells per section to
`157 RN Spp1 Glut`, a midbrain subclass the pruning keeps. None is emitted:
they are `implausible` (a re-mapped cell whose new subclass is not
confident), below their parent or region-dropped.

**T2 after pruning.** In E7's definition (G3's and MO2's) 0–16 cells
(≤ 0.012%) per section stay T2 after pruning. Counting every subclass with
< 25% of its MERFISH cells in the present divisions (the stage C report's
"plan" reading, withdrawn in review) would add never-drop subclasses the
rule never removes (`316 Bergmann NN`, `320 Astro-OLF NN`, `317 Astro-CB
NN`: 178 of 222 such calls on ag7 reseg are `316 Bergmann NN`), nearly
never emitted (2 confident `320 Astro-OLF NN` subclass calls on ag7 reseg,
1 on VZG2 proseg_hybrid).

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
| `scripts/acceptance/resolve_criteria.py` | M4 | The human criteria (H1–H5, H7–H10, H16, H17) re-measured on `merxen annotate-resolve` label tables, with the pre-registered thresholds and the flip rule; H4 on the held-out re-maps, including RESOLVE itself run with the held-out WHB call |
| `scripts/acceptance/marker_pseudo_labels.py` | M4, H9 | Ports of the marker pseudo-label methods of the P7513 (§C) and P1212 (§4) dataset reports, for H9 and H17 |
| `scripts/acceptance/draw_spread.py` | M8, D2 | The draw-spread table of a human self-map bundle (pre-registration §18 item 2): the 3 × 3 grid of per-cell × efficiency seeds (0–2), each draw simulated and mapped on the bundle's own (D1) test cells with the self-map's recorded worker count, decided alone (PREP unweighted, and reweighted to each dataset; `--resolve`: RESOLVE per draw, checked to have loaded the replaced bundle); H7, the H8 warning value, H18 (the panel family's floors), the would-raise list and the validated bins per draw and dataset. The scored draw is the bundle's stored rows (its re-map is the control); the cache is keyed by bundle and worker count; `draw_spread_run.json` records the code commit (git, `MERXEN_CODE_COMMIT` or the export's `COMMIT` file) and module sha256 |
| `scripts/acceptance/mouse_corr_shadow.py` | M6, §7.3 | The mouse class `avg_correlation` floor study (none, 0.40, 0.50) on the ag7 proseg_hybrid and VZG2 original_seg label tables: MO1's check, coverage and marker consistency by depth, and the pre-registered selection (pre-registration §16) |

The results and the decisions they feed (X1, OD-B6 / OD-B7, OD-B8, OD-B13, the
H4 and H16 baselines) are in §11 of the pre-registration document.

## Annotation report (`merxen annotation-report`, M7)

`merxen.annotation.report` builds the QC report of plan §9 from the
published outputs of one human pair × segmentation (or one mouse section ×
segmentation): RESOLVE (`<pair>_resolve_summary.json`, the label tables and
annotation manifests), MAP (`map_manifest.json`; the mouse region step's
`<sid>_mouse_regions.parquet`), PANEL (`panel_report.json`), the clustered
H5ADs of COMPUTE_CPU / FINALIZE (coordinates and counts), cortical depth,
the shared tissue mask and the MENDER manifests when present, and the
bundles the provenance names (read-only). It writes a fresh directory and
never into a results tree unless `--allow-results-output` is given: not
inside an input directory, not at or below the results root of any input
(also for explicit `--resolve-dir` / `--map-dir` / `--panel-dir`, depth
tables and MENDER manifests), not inside the store or a recorded bundle, not
beside the held-out CSV. In map_first
runs the pipeline builds it as `ANNOTATION_REPORT` (below), which publishes
to `<pair>/<seg>/annotation_report/annotation_report_out/`.

| Item (§9) | What the report shows | Metrics (`acceptance_metrics.json`) |
|---|---|---|
| 1 Annotatability, panel | Confident fraction per level of table cells and of segmented objects, status breakdown, resolvable share, gate values and reasons, trust state and basis, realised flag rates per class × platform, the self-thinning eligibility (≥ 500 cells with ≥ 200 counts) with its truth composition over the labelled deep cells, the unlabelled share, the argmax composition, the evaluable classes and a reliability verdict, and the **panel card**: gene-ID resolution by source, unresolved features, controls by type, markers per parent (weak and collapsed parents, nodes MapMyCells patches with ancestor markers), precision-coverage curves per level and depth, D_max and extrapolated share per class, PREP (unweighted) vs RESOLVE (reweighted) emission, thresholds and floors with their source; banners for `provisional`, `broad_only` and `refused` panels and failed or broad-only gates | H7, H8 (MO7), H16, H18 inputs |
| 2 Composition | Soft (headline), soft ≥ 30 counts, confident-only and argmax composition per level with 95% block-bootstrap CIs (500 µm tiles, 200 replicates, seed 0), whole section vs shared mask, per depth bin, `Mixed/Unknown` by reason; the COP-derived part of every OPC bar (`cop_derived_share`, hatched); the sinks table (vocab sinks, region-implausible nodes, COP and CGE, and nodes whose argmax share of their broad class exceeds the reference share of that class more than 3×, with argmax, soft-mass and confident shares); leaf-level rows of a sample whose gate is not full marked `withheld_for_comparison` and drawn hatched; COP control | H2, H5 (with `cop_derived_soft_opc_share`, `cop_derived_argmax_opc_share`; MO4 shares) |
| 3 Confidence vs counts | 2D histograms of raw bp and `avg_correlation` vs counts per platform and level, with the raw thresholds | — |
| 4 Reference expectation | Observed fraction-positive and mean log2(CPM+1) per confident leaf label on its most specific panel genes and the §5.8 canonical markers on the panel, beside `profiles.parquet`; pseudobulk-centroid r and a correlation re-mapping to the nearest reference centroid. A sample whose gate is not full (or without a confident leaf label) is shown on its confident broad labels, with reference profiles aggregated from the supercluster rows (reference-cell-weighted) | report-only |
| 5 Negative-marker purity | `contamination_score` quantiles and `flag_contaminated` rate per confident broad label and platform | report-only |
| 6 Method agreement | WHB vs SEA-AD at 7 classes and lineage by count quartile, consensus tier fractions and a tier map, the below-60 rule's coverage cost (cumulative: cells < 60 counts whose first vote failure, `method_disagree` / `single_method`, is at the level or at lineage; the per-level count beside it) | H3 |
| 7 Cross-platform (human) | Soft broad and supercluster JSD (whole section, shared mask, joint block bootstrap), set c beside set a (soft, soft ≥ 30, argmax and supercluster), per-type density correlation in 200 µm aligned bins, `<pair>_platform_gene_factors.csv` (median-centred per-gene log2 Xenium / MERSCOPE within confident labels) and per-label pseudobulk r; every statement follows RESOLVE's cross-platform scope, withheld ones are recorded as `withheld`. A `per_platform` pair (§8.5) takes all of them from the intersection run (`mmc_whb_xpanel`: H1 with its recomputation and figure; density, factors and pseudobulk r on its argmax broad labels); its own-panel comparisons are `withheld` (`own_panel_not_comparable`) | H1 |
| 8 Held-out genes | The held-out-gene enrichment rows of the pair (`--heldout-csv`: `heldout_genes.py` or `resolve_criteria.py` output); with several label sets, H4's scored set, the non-circular WHB-only held-out re-resolve `m4_resolve_heldout_whb_only` (M8 D6), is shown, else `heldout_whb_confident`, else an argmax set, all sets in the table CSV; a set other than the scored one is noted as not the scored set and its records are criterion `report`, so every H4 record comes from the assigned set; mouse: the spill-over held-out test and astrocyte FPR | H4, MO5 |
| 9 Cortical depth (human) | First the depth input: QC warnings, coordinate source and ribbon share per platform, and a label-free cross-platform check (below); then the median depth per confident supercluster (NP and CT/6b merged) and broad class with 95% CIs over 500 µm tangential blocks (square tiles as sensitivity); the ordering Upper-layer IT < Deep-layer IT < Deep-layer NP/CT/6b with non-overlapping CIs per platform and its replication across platforms; oligodendrocytes WM > GM; astrocyte / neuron / oligodendrocyte depth gradients (shares of confident broad cells) | H12 |
| 10 Mouse regions | Inferred vs explicit regions, the tile map, the drop list, relabelled cells and targets, subclass × region vs MERFISH shares, class composition vs the AP-matched window (shares of the allocated soft mass, as G4, with the unallocated share beside them), Astro-Epen with and without low-count cells, F1 rate, spill-over prevalence and its map, gate G1–G5; the M6b fields (AP estimate and bin, rule variant) are read when present and reported `pending_m6b` otherwise; a section with an AP estimate below 6.0 mm gets the MO11 banner ("MO11 not yet passed") | MO2–MO7, MO10 / MO11 placeholders |
| 11 AD and OOD (human) | Depth-matched real / in-silico `avg_correlation` ratio per class (in-silico = the primary bundle's correct resolvability calls), `flag_ood` rate, SEA-AD disease-supertype share per class (population level, `calibrated = false`) | report-only |
| 12 Provenance | References, bundle and lookup hashes, label sha256, ctm version and commit, seeds, wall times, panel family, trust state, calibration kind, MENDER digest; the H13 id-set check (label `in_table` ids = clustered H5AD ids) | H13, H14 inputs |

**Metrics, not verdicts.** `acceptance_metrics.json` lists every measured
value as a record with a coverage table of the §14 criteria: which come from
the report and which from the acceptance scripts (H6, H9, H10, H15; MO1's own
definition, MO10). Pass / fail against the pre-registered thresholds is the
acceptance programme's (M8, M9). The file, the HTML, the CSVs and the figures
are byte-identical for identical inputs (no timestamps; seeded bootstraps);
wall time and versions go to `report_run.json`.

**The `acceptance_metrics.json` contract** (`report_model.AcceptanceMetrics`,
`REPORT_SCHEMA_VERSION` 1; M8 and M9 read it). Top-level keys:

| Key | Type | Content |
|---|---|---|
| `schema_version` | int | `REPORT_SCHEMA_VERSION` |
| `kind` | str | always `merxen_annotation_report_metrics` |
| `species` | str | `human` or `mouse` |
| `pair_id` | str | pair (mouse: the section id RESOLVE used) |
| `segmentation` | str | segmentation |
| `metrics` | list of records | every measured metric (below), sorted by criterion, name, sample, region, level, kind, group |
| `criteria` | list | one row per §14 criterion of the species: `criterion`, `source` (`report`, `report+script` or `script:<name>`), `description`, `n_records` |
| `datasets` | object | per sample id: platform, counts, trust state and basis, panel family, gate level and warning, banner flag, degraded mode, coordinate source, whether cortical depth was read |
| `items` | object | per item slug: `number`, `title`, `status` (`ok`, `partial`, `not_available`, `not_applicable`, `failed`), `notes`, `figures`, `tables` |
| `provenance` | object | the provenance footer (item 12) |

Every record of `metrics` (`report_model.MetricRecord`) has these fields:

| Field | Type | Content |
|---|---|---|
| `criterion` | str | §14 id (`H1`, `MO6`) or `report` for a report-only metric |
| `name` | str | stable snake-case metric name |
| `scope` | str | `dataset` (one sample) or `pair` |
| `sample_id` | str or null | the sample (`null` for pair metrics) |
| `platform` | str or null | its platform |
| `region` | str or null | `whole_section` / `shared_mask` where relevant |
| `level` | str or null | annotation level where relevant |
| `kind` | str or null | composition kind or variant (`set_a:soft`, `intersection_run:soft`, `square_tile_500um`, `outside_ribbon_proxy`, ...) |
| `group` | str or null | a class, supercluster or stratum |
| `value` | number, bool, str or null | the value (`null` unless `status` is `measured`) |
| `ci_low`, `ci_high` | number or null | 95% CI |
| `n` | int or null | cells (or tiles, bins) behind the value |
| `definition` | str | how it is computed |
| `source` | str | where it comes from (`resolve_summary`, `label_table`, `report_metrics.<function>`, `bundle`, ...) |
| `status` | str | `measured`, `not_available` (the input is missing or invalid; the note says why, e.g. `depth_input_invalid`) or `withheld` (the cross-platform scope forbids the statement, e.g. `own_panel_not_comparable`) |
| `note` | str | anything a reader must know |

Adding, renaming or removing a top-level key or a record field bumps
`REPORT_SCHEMA_VERSION`; `tests/test_annotation/test_report_model.py` pins
both field sets against this table.

**Definitions chosen here (the plan leaves them open).**

- *H12 depth input (pre-registration doc §17):* before any H12 value the
  report checks the depth input without labels. In the 200 µm bins of the
  Xenium frame holding ≥ 5 table cells of each section, the two sections must
  agree on ribbon membership (more than half of a bin's cells inside the
  cortical ribbon) in ≥ 80% of the bins, and their per-bin mean depths must
  correlate with r ≥ 0.8. When they do not, the platform whose depth was
  computed on transformed coordinates (`coordinate_source` naming an
  `_aligned` shape; both platforms when neither or both did) gets every H12
  metric, and the pair its replication, as `not_available` with the reason
  `depth_input_invalid`, an error banner, and the item status `partial`; the
  tables keep the computed values with `depth_input_valid = false`, and the
  figures label that platform "depth input invalid". A single platform, or
  coordinates not in one frame, is `not checked` and stays measured.
  Calibration on the published outputs: the native-frame (reseg) MERSCOPE
  depth of the same cells gives .966 / r .987 (P7513) and .938 / r .978
  (P1212); the published proseg_hybrid MERSCOPE depth, computed on the
  aligned coordinates with native-frame boundaries, .530 / r .144 and
  .473 / r −.114.
- *H12 CIs (pre-registration doc §17):* the bootstrap unit of the medians is
  a 500 µm tangential block, `floor(tangential_position_um / 500)`: a full
  pia-to-WM strip, so every replicate keeps the depth slices in their
  anatomical proportion, and about ten neighbouring streamlines, so the
  blocks are not near-duplicates. The square 500 µm tile bootstrap (§5.5's
  composition unit) is reported as a conservative sensitivity
  (`kind = square_tile_500um`): a tile holds about a fifth of the cortical
  depth, so resampling tiles re-mixes depth slices and widens the interval.
  `depth_ci_method` records the unit. Without tangential positions the tiles
  are the (only) method. *Scored CI (pre-registration §20 D11, approved by
  the user on 2026-10-01; it supersedes the square tiles of §18 item 3):*
  the M8 gate scores the H12 ordering on the tangential blocks
  (`report_depth.SCORED_CI`, `depth_ci_scored`): per platform the kind-less
  `depth_ordering_passes` record (the tiles where a platform has no
  tangential positions) and for the pair the kind-less
  `depth_ordering_replicated` record; `report_scoring.score_h12` scores them
  and reports the square-tile records (`kind = square_tile_500um`, always
  written) beside them. WM > GM keeps its square-tile CI, because white
  matter has no tangential position. For a dataset whose gate is not `full` (no
  supercluster labels) the ordering is `not_available` under both methods,
  and so is the pair's replication (P1212: the MERSCOPE section is
  broad-only).
- *H12 WM > GM:* with a `white_matter` label in the depth output (the brain
  outline annotated) the share is compared between `white_matter` and
  `grey_matter` cells; without it (the current outputs) cells outside the
  cortical ribbon stand in for white matter (`kind = outside_ribbon_proxy`).
- *Replication (H12, "on both platforms"):* the ordering passes on both
  platforms, reported with the Spearman correlation of the supercluster
  median depths between the platforms.
- *Self-thinning (item 1):* eligibility and truth composition only. The
  truth is the full-depth confident label, so the composition is over the
  labelled deep cells, with the unlabelled share and the WHB argmax
  composition beside it; classes with ≥ 50 labelled deep cells are
  evaluable. The run is unreliable when more than 30% of the deep cells
  have no confident label or one class holds ≥ 80% of the labelled ones
  (P1212_MERSCOPE: 48% unlabelled, 83% of the labelled Fibroblasts). The
  thinned re-map itself is v1.1 (§5.4).
- *Sink-prone nodes (item 2):* COP and CGE (the plan's named sinks), and any
  node whose argmax share of its broad class exceeds its reference share of
  that class more than 3×. The comparison is within the broad class because
  the WHB frontal precompute is neuron-enriched (neurons hold 90% of its
  cells), so a share of all reference cells flags every glial node.
- *Pseudobulk re-mapping (item 4):* nearest reference centroid by the
  correlation of log2(CPM+1) profiles, not a MapMyCells run.

```bash
merxen annotation-report --pair P7513 --segmentation proseg_hybrid \
  --results-root /srv/storage/MerXen/results \
  --store /media/mathieubo/SSD1/MerXen/annotation_references \
  --out /path/outside/results/P7513_proseg_hybrid_report
```

On the published P7513 and P1212 proseg_hybrid outputs a report takes
about 40 s and 4 GB (one process).

### `ANNOTATION_REPORT` in the pipeline (M7)

`ANNOTATION_REPORT` (`workflows/modules/annotation.nf`) runs `merxen
annotation-report` once per map_first pair × segmentation (mouse: section ×
segmentation). `ANNOTATION_REPORTING`
(`workflows/subworkflows/annotation_report.nf`), called at `main.nf` hook
H11 in map_first runs with `annotation_report_enabled` (default `true`),
releases a branch once these have finished:

| Waits for | When | Staged as |
|---|---|---|
| RESOLVE, MAP, PANEL (`CLUSTERING_MAP_FIRST.labels`) | always | `report_inputs/annotation_{resolve,map,panel}_out` |
| FINALIZE (the clustered H5ADs) | always | `report_inputs/clustering_squidpy_out` |
| `COMPUTE_CORTICAL_DEPTH` of every active platform of the pair | cortical depth runs after clustering and covers the segmentation | `report_inputs/cortical_depth_<n>/compute_cortical_depth_out`, passed as `--cortical-depth-dir <sample>=<dir>` |
| `MENDER_FINALIZE` of every sample of the branch | MENDER runs for the segmentation | `report_inputs/mender_<n>/<platform>`, passed as `--mender-manifest` |
| the pair's ALIGN files (shared tissue mask) | the pair is aligned | `report_inputs/align_out` |

A branch whose neither depth nor MENDER runs starts as soon as FINALIZE
finishes, and no branch waits for another pair's or segmentation's tasks.
The bundles are read where RESOLVE's manifests say they are (not staged);
RESOLVE's outputs carry their hashes. The task runs on the CPU in the main
environment (`CUDA_VISIBLE_DEVICES=""`, BLAS threads = `task.cpus`) with 4
CPUs and 32 GB (`conf/annotation.config`); on the-dwight at most
`annotation_report_max_forks` (2) run at a time. It never passes `--strict`:
an item that raises is recorded in the report as `failed`, and the task
still publishes. The task hash carries a fingerprint of the report code
(`AnnotationReport.REPORT_SOURCES`: `merxen/annotation`, its packaged
tables, `merxen/clustering`, the command, and `merxen/gene_ids.py`,
`merxen/palette.py` and `merxen/plotting.py`, which shape its outputs from
outside them; `test_annotation_report_module.py` pins what else they import),
so a report change re-runs the report under `-resume`; no other task stages
its output.

**Failure semantics.** A branch whose depth, MENDER or report task fails is
dropped under `errorStrategy "ignore"`, like its MENDER; the end-of-run
summary (hook H6) lists it under "annotation reports not built", and a
report with `failed` items under "annotation report items failed". A
held-out-gene CSV is not produced in the pipeline, so item 8 is
`not_available` there until the acceptance programme (M8) runs
`heldout_genes.py`; standalone builds take it with `--heldout-csv`.

## Known limitations

- **Four WMB subclasses have no 10Xv3 reference cell** (`157 RN Spp1 Glut`,
  `279 PSV Pax2 Gly-Gaba`, `280 NLL-po Pax7 Gaba`, `297 CU-ECU Pax2 Gly-Gaba`;
  sequenced only by 10X Multiome, ≤ 0.2% of cells in any in-scope posterior
  section; user decision 2026-09-26). They are not filled in, but they stay
  in the mapping tree and the Allen means, so `cell_type_mapper` can still
  assign them with their ancestors' markers (an ag7 run with the default,
  not the panel, lookup did: 12 cells as `157 RN Spp1 Glut`; the M6 exit
  runs with the panel lookup call none of the five on ag7 or VZG2, see
  [Mouse exit on real data](#mouse-exit-on-real-data-m6)). Such labels have
  no marker support of their own: each mouse bundle lists these nodes in
  `marker_unsupported_nodes` (and `uncovered_subclasses`,
  `uncovered_clusters`), and RESOLVE makes such a call `not_resolvable` at
  that level (never emitted; the resolve summary counts them). Query cells
  of these types therefore never get these subclass labels: they map to the
  nearest covered subclass or stay unresolved at subclass level. They can
  occur only in posterior sections (AP ≥ 7.5 mm).
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
- **Mouse scope.** The region rule is for coronal sections at AP 2.4–10.4
  mm that are not OB- or CB-dominated. Real data validate only the
  hippocampal / thalamic level (ag7 and VZG2, AP 8.0–8.5 mm); on MERFISH the
  rule's harm is anterior (AP 3.0–6.0 mm: missed OB slivers next to the AON
  drop OB-type interneurons), which M6b scores (plan §7.8). Give sagittal,
  OB- or CB-dominated sections an explicit `mouse_section_regions` or
  `none`.
- **Parts of the mouse gate are provisional** (M6). G2's 0.70 / 0.80
  thresholds were set on MO1's hand-listed marker referee (.911 / .856 on
  ag7 / VZG2), but the gate derives its markers per panel and reads .805 /
  .819 on the same sections [L]. G4 is not evaluated until M6b's AP
  estimate chooses the MERFISH window (standalone runs pass
  `--mouse-g4-sections`). The pipeline's RESOLVE task does not stage the
  QC stage's registration check yet (M5 wiring) and refuses a mouse sample
  without one; a standalone run passes `--registration-qc` or
  `--registration-qc-dir`, and without either G1 is not evaluated and the
  gate warns. G1 is the signal that catches misregistered data: per-cell confidence does not
  (class bootstrap ≥ 0.9 covers .733 of the invalid VZG2 proseg_hybrid cells
  and .739 of ag7 proseg_hybrid's; M6 exit).
- **Mouse flags are report-only QC columns.** `flag_microglial_spillover`
  measures spill-over prevalence, never a microglia count, and a panel with
  fewer than 6 derived microglia genes (ag7: 5) gets no held-out check.
  F1 (`flag_region_incoherent`) catches only about a third of the sink calls
  on good data (VZG2; E7 §3), so it does not replace pruning.
- **No mouse `avg_correlation` floor.** The M6 shadow study selected none
  (pre-registration §16); `ct_class_corr` is recorded only, and the v2
  mouse calibration (plan §7.3) revisits it.
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
  families exist. *Since the M3c follow-up (2026-10-06):* RESOLVE reads them
  (version-7 decisions, `flag_nonneuronal_high_depth` and the class-depth
  prediction at the dataset's own per-class depth); the real-data QC
  warnings (`real_qc`, e.g. the per-class coverage check) are not yet wired
  into RESOLVE or the report (M13). On the public section the version-7 predictions
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
- **RESOLVE's resolvability exceptions remove confident oligodendrocyte-lineage
  labels on the shallow MERSCOPE sections** (M4 shadow run, 2026-09-28;
  `m4/STAGE_D_REPORT.txt`, `m4/CRITERIA_AFTER_M4.txt`). Without the
  resolvability tables RESOLVE reproduces the M3 shadow coverage exactly;
  with the version-6 tables reweighted to each dataset, broad
  Oligodendrocytes are not emitted at 10-60 counts on P5011 MERSCOPE (no
  confident broad Oligodendrocytes at all, against 14.8% of table cells
  without the tables) and at 10 counts on P1212 MERSCOPE (15.8% vs 24.0%),
  and broad OPC at 10-15 counts on every dataset. The cells keep their
  confident lineage (branch `Oligodendrocyte lineage/unresolved`) and the
  soft compositions (`soft_broad_*`, H1) are unaffected, but confident-only
  compositions of these sections undercount the oligodendrocyte lineage.
  Confident broad coverage falls by .093 (P1212 MERSCOPE) and .171 (P5011
  MERSCOPE) against M3, H7 fails on P5011 MERSCOPE (.288 < .30) and P1212
  MERSCOPE gets the gate warning (.120 of segmented objects). In the set a
  self-map the wrong broad Oligodendrocyte calls are COP test cells called
  Oligodendrocyte; scoring those as correct at the broad level (a candidate
  fix for the M8 gate PR, measured but not adopted) restores every broad
  Oligodendrocyte label (P1212 / P5011 MERSCOPE broad coverage .427 / .436).
  A decision for the gate PR (OD-B14).
- **Set c of families without a curated list** uses the label-free rule,
  which drops far more genes than E5's validated set c (44-73 per pair on the
  E5 pairs); treat such set-c results as provisional.

## Licences

ABC Atlas, WHB, WMB, MERFISH-C57BL6J-638850 and SEA-AD data are CC BY-NC 4.0;
`cell_type_mapper` is under the Allen Institute Software License.
