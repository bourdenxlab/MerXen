# Reference-based annotation (in development)

Reference-based cell-type annotation replaces the legacy marker scoring of
[Squidpy clustering](clustering-squidpy.md) when a species runs in
`map_first` mode. It is being built milestone by milestone
(`docs/plans/robust-celltype-annotation-plan.md`); legacy runs are
unchanged. This page covers what exists so far: declared panels
(`merxen annotation-panel`) and the reference bundles that
`merxen annotation-reference-prep` builds into the reference store.

## Reference bundles

A bundle holds everything mapping needs from one reference on one declared
panel. Bundles live at `<store>/<reference_id>/<build_hash>/` and are never
modified or deleted automatically (see [CLI reference](../cli.md#merxen-annotation-reference-prep)).

| Reference | Role | What the builder does |
|---|---|---|
| `whb_frontal_supc_clus` | human primary (also the set-c sensitivity bundle) | Copies and verifies the frontal WHB region precompute (ROIs Human A44-A45, A46, A32, ACC; 653 subclusters, 125,481 cells, 59,357 genes), or rebuilds it from the WHB h5ads; truncates it to supercluster → cluster; finds panel reference and query markers (`n_per_utility` 30). |
| `seaad_mr_panel` | human secondary | Copies the SEA-AD Multiregion precompute (release 20260711, 207 supertypes × 36,601 genes; md5 `9b1d5f50a412cd69311fa1175d4836a7`, sha256 `abbd4984bc45…3f6a`) and its small taxonomy tables; panel markers. Missing pinned files are fetched into `<store>/.downloads` (or seeded from a verified local copy). |
| `wmb_panel` | mouse primary | Samples at most `annotation_wmb_max_cells_per_cluster` (50) cells per cluster from the local WMB-10Xv3 h5ads (seed 1; self-map test cells excluded), restricted to the panel genes; builds the marker precompute; finds panel markers with the supertype level dropped; copies the Allen `precomputed_stats_ABC_revision_230821.h5` (md5 `d13b316a1755c459d75b8e45ff92ddfc`) as the mapping precompute. |
| `wmb_region_share` | mouse, panel-independent | Class and subclass × CCF-division shares of the ABC MERFISH-C57BL6J-638850-CCF cells (OB = MOB, AOB, olfactory nerve layer and OLF-unassigned anterior of AP 2.5 mm, split from OLF), each node's home division, per-section compositions and AP composition windows (each section with its neighbours) for every section. |
| `whb_whole_ctx_panel` | human, optional | Whole-WHB panel markers with the 16 cortex-implausible superclusters dropped and the lookup filtered to the pruned tree; refused above 1,000 genes. |

Every bundle records in `bundle.json` the full sha256 of each source, the
copied files, the `cell_type_mapper` steps (input JSON, logs, wall time and
peak RSS under `logs/`) and these diagnostics: levels and node counts,
markers per parent (min / median), weak parents, parents auto-collapsed for
lack of markers and the leaves they hide, lookup keys dropped because the
mapping tree lacks the node, and the panel coverage (panel genes in and
absent from the reference, root markers, root children separated).

### Bundle files

| File | Content |
|---|---|
| `mapping_precompute.h5` | Precompute MapMyCells maps onto (a checksummed copy or a derived file). |
| `source_precompute.h5` | WHB: the verified copy of the region precompute (subcluster leaves). |
| `marker_precompute.h5`, `marker_training_cells.csv` | WMB: the panel-restricted marker precompute and the cells it was built from. |
| `query_markers.json` | The lookup as `cell_type_mapper` wrote it. |
| `query_markers.filtered.json` | The lookup restricted to the mapping tree and the panel; auto-collapsed parents keep an empty entry. |
| `mapping_tree.json` | The mapping tree (after `drop_level` and `nodes_to_drop`). |
| `profiles.parquet` | Per node × panel gene: detection fraction, mean log2(CPM + 1), expected CPM (conditional log-normal from `sum` / `sumsq` / `gt0`) and its share of the panel. |
| `negative_genes.parquet` | Primary references: per broad class × panel gene, the detection fraction in each reference and whether the gene is negative (< 1% in every reference, not a curated state gene). |
| `vocab_snapshot.csv` | Every mapping-tree node with its vocabulary broad class, NT and flags. |
| `depth_grid.json` | The resolvability depth grid of the panel. |

## Known limitations

- **Four WMB subclasses have no 10Xv3 reference cell** (`157 RN Spp1 Glut`,
  `279 PSV Pax2 Gly-Gaba`, `280 NLL-po Pax7 Gaba`, `297 CU-ECU Pax2 Gly-Gaba`;
  sequenced only by 10X Multiome, ≤ 0.2% of cells in any in-scope posterior
  section). They are not filled in: query cells of these types map to the
  nearest covered subclass or stay unresolved at subclass level. Each mouse
  bundle lists them in `bundle.json` (`uncovered_subclasses`,
  `uncovered_clusters`).
- Auto-collapsed parents are still assigned inside by `cell_type_mapper`
  (with their ancestors' markers); levels below them must not be emitted.

## Licences

ABC Atlas, WHB, WMB, MERFISH-C57BL6J-638850 and SEA-AD data are CC BY-NC 4.0;
`cell_type_mapper` is under the Allen Institute Software License.
