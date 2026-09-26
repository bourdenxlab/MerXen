# Reference-based annotation (in development)

Reference-based cell-type annotation replaces the legacy marker scoring of
[Squidpy clustering](clustering-squidpy.md) when a species runs in
`map_first` mode. It is being built milestone by milestone
(`docs/plans/robust-celltype-annotation-plan.md`); legacy runs are
unchanged. This page covers what exists so far: declared panels
(`merxen annotation-panel`), the reference bundles that
`merxen annotation-reference-prep` builds into the reference store, and the
two pipeline processes that run them (`--annotation_prepare_only`).

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

Not in it, and recorded in `bundle.json` instead: the panel's symbols
(bundles hold Ensembl IDs only and match state genes by ID, so a gene list
and a prepared pair panel with the same IDs share one bundle, whatever
symbols they declare; `built_from_panel` names the panel that first built
it), the MAP bootstrap settings, and the resolvability settings
(`recorded_settings`; no M2 bundle holds resolvability output, and threshold,
trust and flag changes re-run only RESOLVE).

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

## Pipeline processes

Two CPU processes in `workflows/modules/annotation.nf`, wired by
`workflows/subworkflows/annotation_references.nf`; neither takes the GPU
lock, and a default (legacy) run instantiates neither.

| Process | Runs | Resources | What it does |
|---|---|---|---|
| `ANNOTATE_PANEL` | once per pair × segmentation | 1 CPU, 4 GB | `merxen annotation-panel` on the gene list (`--annotation_panel_genes_path`) or on the pair's prepared H5ADs; writes the declared panels and `required_bundles.json`. |
| `ANNOTATE_REFERENCE_PREP` | once per unique (species, reference, `panel_hash`) across the run | 8 CPUs, 64 GB, 8 h; above 1,000 panel genes `annotation_prep_large_memory` and 24 h; one at a time on dwight | `merxen annotation-reference-prep`: gets the bundle from the store or builds it, and writes `bundle_ref.json`. Seconds when the bundle exists. |

PREP has no `storeDir`: the store's own lock, temporary build directory and
atomic rename keep concurrent launches safe, and its `build_hash` (sources,
`cell_type_mapper` version, builder settings) is only known inside the task.
PREP is never cached (`cache false`): the task hash cannot see the builder
code or the content of its source files, so a cached task could hand MAP a
stale bundle after a builder fix. A re-run takes seconds when the bundle
exists, and `bundle_ref.json` holds only the bundle's identity (whether it
was reused is logged), so an unchanged bundle gives byte-identical output
and MAP's cache holds. Which pair's copy of a shared panel file reaches PREP
first does not matter: the bundle depends only on the panel's IDs.
Each pair × segmentation is then released with exactly the bundle refs its
`required_bundles.json` lists (a same-panel human pair on `proseg_hybrid`
needs three: WHB and SEA-AD on set a, WHB on set c; a `per_platform` pair
five; a mouse section two), as soon as its last bundle is ready, so pairs
with different panels never wait for each other. A pair whose panels are
refused is released with no bundle; one whose PREP failed is dropped.

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
- **Set c of families without a curated list** uses the label-free rule,
  which drops far more genes than E5's validated set c (44-73 per pair on the
  E5 pairs); treat such set-c results as provisional.

## Licences

ABC Atlas, WHB, WMB, MERFISH-C57BL6J-638850 and SEA-AD data are CC BY-NC 4.0;
`cell_type_mapper` is under the Allen Institute Software License.
