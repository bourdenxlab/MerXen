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
| `whb_frontal_supc_clus_ho` | human, resolvability (M3b) | The frontal WHB cells restricted to the panel, without the held-out donor (`annotation_calibration_holdout_donor`, `auto` = the donor with the fewest frontal cells, H19.30.002); clusters with ≥ 5 training cells; a supercluster → cluster → subcluster precompute of the training cells truncated to supercluster → cluster, with panel markers. The held-out donor's cells (≤ 1,000 per supercluster, all cells of rare ones, ≤ `n_test_cells`) are the test set (`test_cells.h5ad`, native panel counts and truth). E2's HO recipe. |
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
   bundle being built.
2. **Simulation.** Every test cell whose native panel counts reach a grid
   depth `D` is thinned binomially to `D` (human panels up to 1,000 genes
   `[10, 15, 30, 60, 120, 250]`; mouse and larger panels
   `[10, 20, 50, 100, 250, 500, 1000, 2000]`); cells never go up. Recipe
   `R1_contam_HO` (E2): per-gene efficiency LogNormal(0, 0.8) (one draw per
   self-map, median 1) inside the thinning, plus 25% of `D` thinned from a
   random cell of another broad class; truth stays the host's. A clean
   (thinning-only) run is the upper bound. Seed 0.
3. **Mapping** with the production configuration (bootstrap factor 0.5, 100
   iterations, seed 0, raw normalisation, the bundle's lookup).
4. **Levels.** WHB: lineage, broad, NT (group probabilities summed over
   same-group runner-ups), supercluster and the report-only cluster; SEA-AD:
   its 7-class call (E2 definition) against the WHB truth; WMB: broad, class,
   NT, subclass and the report-only supertype (cluster calls aggregated to
   their supertype). Calls are keyed by the called node's class: the E2
   floor classes for human (neurons by NT, COP apart at supercluster level),
   WMB class names for mouse. Calls to sinks and region-implausible WHB
   superclusters are excluded (never confident in production).
5. **Local threshold rule** (E2 verdict 3; never the set-level rule): per
   (level, class, depth bin) an isotonic map of correctness on bp is fitted
   on one split half of the test cells (`crc32(cell id) mod 2`); the
   threshold `t*` is the lowest bp ≥ the v1 raw threshold at which it
   reaches the target (capped at 0.99), so thresholds only ever rise. The
   other half checks it: a bin is emitted with ≥ 50 confident calls, a
   Wilson 95% lower bound of their precision ≥ target − 0.02 and coverage
   ≥ 0.2.
6. **`D_max` and extrapolation.** `D_max(c)` is the deepest grid depth with
   ≥ 50 test cells of class `c`; deeper bins inherit its decision and are
   marked `resolvability_extrapolated`. A class without such a bin is never
   emitted (no pooling across classes). Everything not emitted is
   `not_resolvable`; the report-only fine levels are emitted only with
   `annotation_allow_fine_levels` and a seed-1 re-map that changes ≤ 2% of
   their confident labels (`level_emission`).
7. **Regimes.** `validated` (real-data-validated families: base targets,
   the pre-registered default applied, local thresholds that would raise
   it listed for H18), `provisional` (other panels and simulation-validated
   families: targets + 0.05, + 0.10 below 60 counts, capped at 0.97; `t*`
   applied) and `trust` (base targets with the local rule).
8. **Floors** are the smallest emitted depth; for panels without real-data
   validation the applied floor is max(known floor, simulated floor), and
   at least 60 for the mouse subclass level.
9. **Trust constraint:** `refused` when the broad level is emitted for no
   class at any depth ≤ 250 (`trust` regime), `broad_only` when the leaf
   level is emitted for fewer than half of the classes with `D_max` at
   every such depth. The constraint becomes the bundle's `panel_trust`;
   otherwise the panel diagnostics and `validated_panels.csv` decide
   between `provisional` and `validated`.

PREP's decisions are unweighted and set only the trust constraint. RESOLVE
(M4) re-derives them with the simulated cells reweighted to each dataset's
composition (`ResolvabilityTables.decisions(composition=...)`,
`composition_weights`), with Wilson bounds on the Kish effective n.

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
resolvability constraint. Re-deriving the decisions reweighted to a
dataset's composition takes 2–4 s. Applied to the M3 shadow MAP runs of the
eight human datasets (reweighted, validated regime), broad is emitted for
59–82% of table cells and supercluster for 59–83% (0–13% of them in bins
inherited from `D_max`); ag7 proseg_hybrid: class 86%, subclass 85%.
The H18 exceptions are listed under Known limitations.

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
| `resolvability_summary.json` | Recipes, levels, settings, test-set counts, `D_max` per level and class, emission per regime, level, class and depth (status, threshold, `t*`, extrapolated, reason), floors, validated thresholds the local rule would raise, the trust constraint, fine-level seed stability, runtimes and mapping runs. |
| `test_cells.h5ad`, `test_cells.parquet` | Resolvability test-set bundles: native panel counts of the test cells and their truth per level (`truth__<level>`), composition key and spill group. |

### Declared panels, gene IDs and controls (M3b)

`merxen annotation-panel` (and MAP) read each platform's **declared** panel:
a vendor panel file (Xenium `gene_panel.json`, a MERSCOPE codebook, a gene
table) or the unfiltered `var` of the prepared H5AD. A published clustered
H5AD declares the features its control filter recorded
(`uns["merxen_clustering_squidpy"]["control_feature_filter"]`: kept and
removed), not the `min_cells`-filtered `var`, so zero-count probes and
`min_cells` filtering never change `panel_hash` (measured: the P7513 /
P1212 and ag7 declared hashes equal the M3 shadow ones).

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
with a reason). Features resolving to one ID are merged and summed at MAP;
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
drift and the sha256 of the resolution table.

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
its own family. A Xenium panel never inherits a MERSCOPE family.
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
families never do. Measured on the current panels (`m3b/diagnostics/`
in the evidence archive): set a (WHB and SEA-AD bundles) and set c, ag7 and
VZG2 are `validated` (`real_data`); their bundles without a self-map carry
the note `resolvability_not_run` (H18 cannot be checked on them); the
P5011 per-platform panels (268 / 298 genes) preview as `provisional`; ag7
symbols run as human are `refused` (`gene_ids:species_mismatch`).

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
gene list, the references' source params, the region and that the stores are
writable. A later `map_first` run reuses a bundle whenever its declared
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
   (`subset_panels/<sid>_<run_id>.panel_genes.json`) and maps with the
   store's bundle on it if one exists (`merxen annotate` looks in the
   store); otherwise it maps with the restricted lookup and records the
   request (`subset_bundle` in the run record). Build the requested bundle
   with `merxen annotation-reference-prep --panel-genes <subset panel>`.
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
| `map_manifest.json` | The run record above, with `panel_status` (`ok`, or `refused` with `panel_reasons` and no runs) and, per run that needs one, `subset_bundle` (trigger, `used` or `requested`, subset and parent hashes); `annotation-store prune` counts its `build_hash` values as references. |
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
- **H18 does not pass as written on the validated panels** (M3b self-map;
  a decision for the gate PRs). The §8.3 rule needs ≥ 50 confident calls in
  the check half of a bin, while `D_max` needs only ≥ 50 test cells of the
  class, so a class with fewer than about 100–150 test cells at a depth can
  never be emitted there, and the bins beyond `D_max` inherit that
  undecided bin. Set a: broad and supercluster Oligo are not emitted at 120
  and 250 counts (25 confident check-half calls at 120, all correct), and
  broad OPC, reweighted to any of the eight shadow datasets, is not emitted
  at 15–60 counts (Oligo and neuron cells with spill called OPC; the local
  rule would raise 0.73 to 0.85–0.94).
  Mouse (≤ 10 test cells per supertype): `21 MB Dopa` and `22 MB-HB Sero`
  (and on ag7 `28 CB GABA`) are never emitted at class or subclass level,
  and the glial classes stop below their `D_max` (Immune class up to 50
  counts on ag7, 100 on VZG2), so the mouse H18 rule holds for 22 / 29
  (ag7) and 21 / 29 (VZG2) classes with ≥ 50 test cells. Validated thresholds the local rule would raise are listed per
  (level, class, depth) in `resolvability_summary.json`
  (`validated_thresholds_would_raise`). Human Vascular and Fibroblast cells
  (23 and 5 test cells in the held-out donor) are never emitted. On the M3
  shadow datasets the emission table lowers the confident broad share of
  table cells by 8–22 points against the raw thresholds alone (P7513
  MERSCOPE .711 → .495; H7 asks ≥ .62), mostly Oligo cells at ≥ 120
  counts and Vascular / Fibroblast cells.
- **The validated-regime floor source in `resolvability_summary.json`** is
  labelled `real_e2` for every species; the packaged mouse floors are
  `design_v1`. RESOLVE takes the source from the trust decision
  (`TrustDecision.floor_source`), not from the summary.
- **Set c of families without a curated list** uses the label-free rule,
  which drops far more genes than E5's validated set c (44-73 per pair on the
  E5 pairs); treat such set-c results as provisional.

## Licences

ABC Atlas, WHB, WMB, MERFISH-C57BL6J-638850 and SEA-AD data are CC BY-NC 4.0;
`cell_type_mapper` is under the Allen Institute Software License.
