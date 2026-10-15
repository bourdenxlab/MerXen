# Annotation v1 pre-registration: acceptance thresholds and M3 baselines

- **Names and paths (dated note, 2026-10-15):** under the privacy rule of 2026-10-14 (CLAUDE.md), this document writes every sample name as its alias (HD1–HD4, HD1_IF, HD3_IF, HM1–HM4, MS1, MS2; `docs/local-configuration.md`), every host path as its `.env` variable (`$MERXEN_...`) and a bare disk root in words (the storage RAID). This is aliasing only: no criterion, threshold, label, reading, verdict or number changed, and every quotation keeps its words. Inside quotations, names and paths in square brackets are aliases or `.env` variables standing for the originals (privacy rule, CLAUDE.md); the originals are kept in the private records. §9 carries the aliases exactly as the plan's §14 it copies. An evidence path that contains an alias names a directory or file that carries the original name, and a sha256 of a tracked document recorded before this note (such as the plan's, below) was taken on its original text.
- **Effort:** robust cell-type annotation (`robust-celltype-annotation`), milestone M3 (plan §12 "M3", items 1–7 of the shadow programme; items 2–7 are in §11).
- **Plan:** `docs/plans/robust-celltype-annotation-plan.md` rev3; the thresholds in §9 below are its §14, copied verbatim (plan file sha256 `14804ff24dcd46ab7d649775bb25aba1b3bac08d533ddd93d457a87f68895024`, last changed in `06a04a1`).
- **Measured:** 2026-09-27, branch `feature/rca-m3-mmc-engine` (MAP engine `2f346b9`, `96f2834`; shadow metrics `b4a71fe`; baseline script `ed373e5`; items 2–7: LL (vii) `89266ff`, shadow metrics `0afc4d6`, scripts `a6b3839`). **Re-measured after the M3 review** with `2a3f183`, `92d9cd9` and `29d1660`: one tile grid resampled jointly for both sections (H1 CIs), E2's SEA-AD probability definitions, a float32 tolerance on every threshold test, the H4 COP-rule option and both OD-B8 readings. The launcher log of every rerun records the git commit and the script's sha256 (`$A/shadow/<item>/PROGRESS.txt`). The pre-review outputs are in `$A/shadow/superseded_20260927_pre_review/`.
- **Commit SHAs:** the repository history was rewritten on 2026-10-01 (one old co-author trailer removed); every SHA cited here before that date is pre-rewrite. Map it with `$A/GIT_REWRITE_20261001_commit_map.txt` (old new, one pair per line; `$A/GIT_REWRITE_20261001_README.txt` beside it).
- **Evidence root** (`$A`): `$MERXEN_EVIDENCE_ROOT`. The item-1 baselines are in `$A/shadow/baselines/metrics/*.csv` (§10 lists the files); items 2–7 are in `$A/shadow/{e8,x1,heldout,flags,ll,glial_jsd}/` (§11), summarised in `$A/shadow/SHADOW_SUMMARY.txt`.
- **M3c addendum (2026-09-28):** §14 pre-registers the acceptance tests of milestone M3c (Xenium Prime 5K measured efficiency, per-class depth and emission stability; plan §12 "M3c", as of `ce15cde`) before any M3c code, with your decisions of 2026-09-28 and the M3c scope decision. §9 stays the plan's §14 as of `06a04a1`; the plan's later "Version-7 families" paragraph is quoted in §14.1.
- **M3c amendment (2026-09-29):** §15 records the orchestrator decisions D1–D7 of 2026-09-29, **pending your confirmation at PR review**, made after §14 (iii) failed in stage D: eight ensemble members, a one-standard-error margin on E2's spread route, the re-test of (iii) on two new keyed ensembles (stop if it fails), and the per-class warning, lung-scenario headline and panel-card changes. It is committed before any code of the amendment and before any new run; the stage-D failure stays on record (§15.1).
- **M13 (2026-10-06):** §23 pre-registers the real-data acceptance of the new-panel human MERSCOPE family (HM1, HM2, HM3, HM4) and the gate-P definitions that apply to every family, before any measurement on the family. It records the user's decisions of 2026-10-02 (D1 (a), D2 (d), D3 (a), the same-species dry-run rule D4, D25, D26) and marks every item that depends on an unanswered decision as proposed.
- **Mouse decisions (2026-10-14):** the user's decisions of 2026-10-14 are recorded before any gate M score is read: §24's amendment (U1, the T2 reading of the M6b selection and of MO10), §25 (gate M's scoring protocol, U3 and U4, with the orchestrator's readings and MS1's G1 recipe) and §26 (the M13b readings, U7, under the gate-P criteria as revised by §23.21 and §23.22).
- **Afternoon records (2026-10-14, before any gate M run):** §25.1.1 records the user's written MO8 exception for the PREP memory peak, §25.2.1 corrects the scored segmentation of one mouse section (a tightening), and §25.3.1 records that the in-place run may proceed and names gate M's region-share bundle. These records name the two mouse sections by their aliases, MS1 and MS2 (§25.1.1, "Names").

## 1. Rules

1. **The thresholds are fixed by this document.** They are the plan's §14 values, copied verbatim in §9. No threshold is changed here.
2. **Baselines may only tighten a threshold.** A measured baseline can justify a stricter threshold, decided in the M3 PR. Loosening a threshold needs the user's written approval in the gate PR. That includes redefining a metric so that it becomes easier to pass, or dropping a dataset, a class or a criterion. This applies to gate H (M8), gate M (M9) and gate P.
3. **Held-out validation sets are reported separately and are not used for tuning:** the donors HD3 and HD4 (§4) and the second segmentation, reseg (§5; H17 is scored on HD1 and HD2 reseg; HD3 and HD4 reseg are reported for information). The development datasets are HD1 and HD2 on proseg_hybrid (§3). Held-out results never select a rule variant, threshold or floor. The pre-registered selection rules (MO10's variant at M6b, gate P's evaluation rules) are the only exceptions. **Nothing in M3 was tuned on the donors, but E2 was:** the raw thresholds, the packaged floors, the SEA-AD subclass thresholds and the donors' own H1 and H7 thresholds were derived with HD3 and HD4 data (§4, first table). H1 and H7 on the donors therefore test consistency with their own E2 measurement, not out-of-sample performance.
4. **Metric definitions are the ones in §2.** RESOLVE (M4) and the acceptance scripts (M8) must compute the gate metrics the same way or document the difference in the gate PR. A difference that makes a criterion easier to pass counts as a loosening (rule 2).

## 2. What was measured, and how

**Inputs (read-only).** These are the published `<pair>_<PLAT>_clustered.h5ad` of HD1, HD2, HD3 and HD4 × MERSCOPE / Xenium, on proseg_hybrid and reseg: `$MERXEN_RESULTS_ROOT/<pair>/<seg>/clustering_squidpy/clustering_squidpy_out/<plat>/`. The runs use `layers["counts"]` on the table cells (every object in the clustered file; `total_counts >= 10` after control removal). The shared tissue mask comes from `results/<pair>/alignment/align_out/shared_tissue_mask.npy`, in the Xenium frame of `registration_summary.json`. The MERSCOPE coordinates are already in that frame (shape key `*_aligned_nonrigid`). The segmented-object counts come from `$A/research/lowcount/qc_summary.csv`.

**MAP runs (M3 stage A code).** Each pair × segmentation ran `merxen annotate --species human --from-clustered-h5ad <M> --from-clustered-h5ad <X> --store $MERXEN_LEGACY_REFERENCE_STORE --gene-id-fallback-csv <WHB-10Xv3-Nonneurons-raw.h5ad> --n-processors 6 --out $A/shadow/baselines/runs/<pair>/<seg>`, with at most three jobs at once. The engine settings were bootstrap factor 0.5, 100 iterations, seed 0, 6 processes, OMP / OpenBLAS / MKL / numba threads 1, raw normalisation, and cell_type_mapper 1.7.2 (`824caef`).
- The bundles were builder v2: WHB frontal set a `b70dc181` (panel `6e5fd5fb`, 297 genes, post-M0e), WHB frontal set c `f20d11b0` (panel `77bd80ed`, 265 genes, curated E5 family; proseg_hybrid only) and SEA-AD Multiregion `3973770c` (set a).
- **Declared panel for reseg.** The published reseg MERSCOPE `var` of HD2 (299 features) and HD4 (268) is `min_cells`-filtered. A panel derived from it would get a new hash that no bundle has. Plan §3.2 requires the declared panel instead. Those two pairs were therefore re-run through `merxen annotation-panel --panel-file <PLAT>=<same section's proseg_hybrid clustered H5AD>`: that file's unfiltered 300-feature `var` is the declared panel. The re-run gave set a `6e5fd5fb`, and `merxen annotate --panel-dir` then mapped the genes present. HD2 reseg MERSCOPE lacks 1 of the 297 genes. HD4 reseg MERSCOPE lacks 32 of 297 (10.8%). Both were mapped with a restricted lookup (M3 behaviour; §7 item 4).

**Shadow v1 rules** (`merxen.annotation.shadow.evaluate_human_rules`). RESOLVE is M4, so the confident statuses come from a shadow evaluation of plan §5.2 rules 1, 2 and 4 and the §5.4 gate. It uses the raw v1 thresholds (WHB broad / lineage 0.73, supercluster 0.69, SEA-AD broad 0.68) and the packaged set-a floors (`floors_human.csv`). Probabilities are stored as float32 (tidy parquet, `ct_*_raw`), so every threshold test allows 1e-6 below the threshold (`schema.meets_threshold`); without it a bootstrap probability of exactly 0.69 fails 0.69 (0.7–1.3% of table cells sit exactly at the supercluster threshold). There are three rules:
- **Lineage** needs aggregated bp >= 0.73, a plausible node (an implausible node keeps its lineage when SEA-AD agrees at lineage) and the second vote.
- **Broad** needs a confident lineage, aggregated bp >= 0.73, counts >= the class × platform broad floor, and the SEA-AD rule. Below 60 counts SEA-AD must agree at the 7-class level. From 60 counts SEA-AD must not confidently disagree. The COP rule applies: a COP call is broad OPC only with >= 120 counts and supercluster bp >= 0.69, or with a confident SEA-AD OPC call.
- **Supercluster** needs a confident broad, gate level `full`, bp >= 0.69 and counts >= the supercluster floor.

The rules have no resolvability (M3b) and no flags. M4's exit requires RESOLVE's coverage to be within ±0.05 of these baselines.

**Metric definitions.**

| Criterion | Operational definition (code) |
|---|---|
| H1 | Soft broad composition: per table cell, the bootstrap probability of the assigned WHB supercluster and of its 5 runner-ups, aggregated to the 7 broad classes through `whb_supercluster_vocab.csv`. Sinks and nodes outside the 7 classes, plus the residual 1 − Σ, are "unallocated" (`soft_matrix_from_provisional`). The composition is the sum over cells. The JSD is the base-2 Jensen-Shannon **distance** (as `scipy.spatial.distance.jensenshannon` and E1 `real_jsd.csv`) of the renormalised 7-class vectors, MERSCOPE vs Xenium. The 95% CI is a spatial block bootstrap over **matched tile locations**: both sections are tiled on one grid of 500 µm squares in the shared Xenium frame (the MERSCOPE `*_aligned_nonrigid` coordinates; inside `shared_tissue_mask` for the mask variant), and each of 200 replicates draws the tile locations (non-empty in either section) with replacement and applies the same multinomial weights to both sections (seed 0, percentile interval; `shared_tile_codes`, `tile_sums`, `block_bootstrap_jsd(resampling="joint")`). Paired differences of two labellings reuse the same draw for both labellings (`paired_block_bootstrap_jsd_difference`). The interval therefore reflects which tissue is sampled, not anatomy mismatched between the adjacent sections. Resampling each section's tiles independently is reported as a sensitivity only (`ci_*_independent` in `jsd.csv`; it was the pre-review method and widens the HD1 interval from [0.116, 0.143] to [0.095, 0.166]). It is reported on the whole section and inside the shared tissue mask. Also reported: argmax (broad class of the assigned supercluster), confident-only (the v1 shadow confident broad label), the soft composition over cells with >= 30 counts, and set c. |
| SEA-AD probabilities | E2's definitions (`exp/E2/build_tables.py`), on which the v1 SEA-AD thresholds were derived. **Broad** (`seaad_broad_calls(class_level=…)`; the 0.68 `seaad_broad` threshold): the label is the assigned subclass's broad class (vocab; "VLMC & Perivascular" split by the assigned supertype); neurons take the SEA-AD class-level bootstrap probability, every other class takes class bp × (subclass bp + same-class runner-up subclass bp), a split subclass also × its same-class supertype mass. **Subclass** (the 0.55 / 0.45 thresholds, split at `second_vote_below_counts` = 60): the subclass `aggregate_probability` (class bp × subclass bp). **Soft SEA-AD composition** (§11.6): class bp × the subclass-level soft mass. E1's definition (subclass mass alone) is kept for comparison only; it calls 6–19 points more table cells SEA-AD-confident (≥ 0.68; proseg_hybrid) and changes the v1 broad coverage by ≤ 0.04 points. |
| H2 | Share of table cells whose assigned WHB supercluster is a sink (Miscellaneous, Splatter) or not region-plausible for frontal cortex (vocab), whatever its bp. This is `flag_implausible` with `implausible` status taking precedence over `low_confidence`. |
| H3 | Share of table cells with >= 20 counts where WHB's 7-class label (broad class of the assigned supercluster) equals SEA-AD's (subclass → broad class through `seaad_mr_subclass_vocab.csv`, "VLMC & Perivascular" split by supertype). As in E1, two labels outside the 7 classes agree; the strict variant, where they disagree, is in `sample_metrics.csv` and differs by < 0.001 on every dataset. |
| H5 | Confident COP supercluster share and confident broad OPC share of table cells (v1 shadow rules). The COP-derived OPC share is the share of confident broad OPC cells whose WHB call is COP. |
| H7 | Confident broad coverage of table cells under the v1 shadow rules. Coverage of segmented objects is also reported. |
| H8 | Gate on table cells: A = share with >= 30 counts; `broad_only` if A < 0.30; `failed` if confident broad coverage of table cells < 0.25; warning if coverage of segmented objects < 0.15 (`dataset_gate`). |
| H10 | E1 marker referee (`exp/E1/09_marker_referee.py`; `marker_class_scores`, `marker_referee`). For each disputed cell, where both labels are among the 7 classes and they differ, the label whose canonical panel markers (E1's list) hold the larger fraction of the cell's counts wins. "New" is the v1 shadow confident broad label, so cells that are not confident have no new label. "Legacy" is the published `obs["broad_class"]`. |
| H4 | Held-out-gene enrichment (§5.8; `scripts/acceptance/heldout_genes.py`). Markers: `heldout_markers_human.csv` in `rank` order (rank 1 = the plan's list, rank 2 = canonical alternates for markers the panel lacks), only if on the panel, never SST, GAD2 and P2RY12 not on MERSCOPE; 2–3 per broad class (Neurons pool the excitatory and inhibitory markers); a class with < 2 is skipped and counts as not passing. They are removed from the query and the lookup, and WHB alone is re-mapped with the production engine configuration. Per class and platform, over all table cells of the re-map labelled with one of the 7 classes: fold = held-out counts per count in the cells assigned the class ÷ the same in cells assigned another class (pooled); AUROC of the per-cell held-out fraction, assigned vs other. A class passes with fold ≥ 3 and AUROC ≥ 0.70. **"The assigned class" is not defined by plan §5.8 and is an open choice for the user** (§11.3): (a) the re-map's argmax (the M3 working metric); (b) the argmax with the §5.2 COP rule applied (COP calls that fail it stay at lineage and leave the OPC class; with or without the SEA-AD rescue); (c) the WHB-only confident calls. The production labels are reported for comparison (circular). |
| H16 | Realised rates per flag × class × platform over confident broad calls (`scripts/acceptance/shadow_flags.py`, prototype of §5.6). Contamination: counts on the assigned class's negative genes (bundle `negative_genes.parquet`) / total counts; null = beta-binomial MLE on the class's confident cells in its top depth quartile (≥ 30 cells, else no flag and the stratum is uninformative); flag at upper-tail p < 0.01 with ≥ 3 negative counts. Diffuse: distinct query genes > q95 of 200 multinomial draws from the class profile at the cell's query depth. Strata above 15% are marked uninformative (H16); §4.3's 30% for the diffuse flag is reported beside it. |
| OD-B13 | On E2's 30k native cells per dataset (`$A/exp/E2/out/cells_native.csv.gz`, the only cells with likelihood-typer calls; all 30k shared with the table), the confident broad coverage is computed with the same production WHB and SEA-AD calls under five variants: (a) v1 without the below-60 rule; (b) v1; (c) v1 with the likelihood typer instead of SEA-AD as the below-60 vote; (d) E2's LL rule alone (`exp/E2/final_coverage.py`: LL must agree below 60 counts, neurons merged, fibroblasts pooled with vascular cells; no SEA-AD role); (e) no second method. The below-60 SEA-AD cost is (a) − (b). The trigger (> 0.08 on any dataset moves LL before M8) is scored on (c) − (b) and on (d) − (b). |
| M3 exit | Production vs the E1 (iii) pilot (`$A/research/pilot/{whb_region,seaad_mr}`, HD1 and HD2 30k subsets). The comparison covers the WHB supercluster assignment on cells with production bp >= 0.8 (target >= 0.98) and the SEA-AD subclass assignment on all shared cells (target >= 0.945 − 0.02 = 0.925). HD3 and HD4 are compared with E2's runs of the pilot configuration (`$A/exp/E2/mmc/{WHBF,SEAAD}__<sid>_sub30k.json`), for information. |

**Code.** Metrics: `src/merxen/annotation/shadow.py` (tests: `tests/test_annotation/test_shadow.py`). Driver: `scripts/acceptance/shadow_baselines.py --runs-root $A/shadow/baselines/runs --results-root $MERXEN_RESULTS_ROOT --evidence-root $A --out $A/shadow/baselines/metrics` (86 s, 3.5 GB; the shadow scripts take the gene-ID fallback tables from `--gene-id-fallback-csv` or `$MERXEN_GENE_ID_FALLBACK_CSV`).

## 3. Development baselines: HD1 and HD2, proseg_hybrid

These are the rows that the §9 human flip rule scores for H1–H3, H5, H7–H10 and H12–H16.

| Pair | H1 threshold | Soft, whole section [95% CI] | Soft, shared mask [95% CI] | Argmax, whole | Confident, whole | Soft, cells >= 30 counts | Set c soft / argmax | Unallocated soft mass M / X |
|---|---|---|---|---|---|---|---|---|
| HD1 | <= 0.17 | 0.129 [0.116, 0.143] | 0.128 [0.121, 0.134] | 0.130 [0.118, 0.143] | 0.145 [0.130, 0.163] | 0.211 | 0.121 / 0.121 | 0.017 / 0.017 |
| HD2 | <= 0.17 | 0.133 [0.126, 0.142] | 0.154 [0.148, 0.160] | 0.149 [0.138, 0.161] | 0.206 [0.196, 0.220] | 0.183 | 0.109 / 0.117 | 0.019 / 0.020 |

| Sample | Table cells | Median counts | A (share >= 30 counts) | Gate level | Warning | Confident broad, table cells [H7 threshold] | Confident broad, segmented objects | Lineage | Supercluster | Implausible calls (H2 <= 1%) | WHB-SEA 7-class, >= 20 counts [H3 threshold] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| HD1_MERSCOPE | 164,370 | 71 | 0.759 | full | no | 0.704 [>= 0.62] | 0.546 | 0.745 | 0.620 | 0.60% | 0.926 [>= 0.90] |
| HD1_XENIUM | 132,489 | 43 | 0.641 | full | no | 0.597 [>= 0.44] | 0.471 | 0.640 | 0.519 | 0.35% | 0.917 [>= 0.90] |
| HD2_MERSCOPE | 102,886 | 17 | 0.234 | broad_only | no | 0.438 [>= 0.34] | 0.152 | 0.598 | 0.000 | 0.30% | 0.825 [>= 0.80] |
| HD2_XENIUM | 157,676 | 25 | 0.426 | full | no | 0.510 [>= 0.29] | 0.332 | 0.589 | 0.446 | 0.36% | 0.908 [>= 0.80] |

| Sample | COP argmax calls | Confident COP supercluster (H5 <= 2%) | Confident broad OPC (H5 <= 10%) | COP-derived share of confident OPC | COP suppressed (`flag_cop_suppressed`) | New vs legacy referee: disputes (share); new / legacy / tie (H10 >= 0.70) | WHB vs SEA-AD referee: disputes (share); WHB / SEA-AD / tie |
|---|---|---|---|---|---|---|---|
| HD1_MERSCOPE | 4.14% | 0.28% | 2.32% | 0.620 | 0.86% | 50,969 (0.310); 0.989 / 0.005 / 0.006 | 17,683 (0.108); 0.395 / 0.339 / 0.267 |
| HD1_XENIUM | 2.80% | 0.12% | 1.51% | 0.623 | 0.54% | 20,677 (0.156); 0.980 / 0.007 / 0.012 | 13,667 (0.103); 0.466 / 0.370 / 0.164 |
| HD2_MERSCOPE | 8.90% | 0.00% | 1.43% | 0.823 | 3.50% | 43,548 (0.423); 0.961 / 0.006 / 0.032 | 22,384 (0.218); 0.298 / 0.314 / 0.388 |
| HD2_XENIUM | 4.18% | 0.02% | 1.14% | 0.691 | 1.08% | 24,576 (0.156); 0.978 / 0.008 / 0.014 | 19,822 (0.126); 0.369 / 0.365 / 0.266 |

- **H1.** The soft JSD is 0.129 (HD1) and 0.133 (HD2), against a threshold of 0.17. The plan's basis (E2 `cells_native`) was 0.139 / 0.129. The argmax JSD is 0.130 / 0.149, against E1 `real_jsd.csv` (iii) 0.132 / 0.143. With matched tiles the intervals are 0.116–0.143 and 0.126–0.142. The pre-review intervals (0.095–0.166 and 0.122–0.154) resampled each section's tiles independently, which mismatches anatomy between the adjacent sections (extra white matter on one side, extra grey matter on the other); their width was that artefact, not tile heterogeneity (§2). Confident-only compositions are further apart (0.145 / 0.206; the draft floors gave 0.236 / 0.285), which is why H1 is soft (D-C5).
- **H2.** The implausible share is 0.30–0.60%, within the 1% threshold (E1 (iii): 0.3–0.6%). The calls go to Amygdala excitatory (region-implausible), Miscellaneous and Splatter (`implausible_top_nodes`).
- **H3.** Agreement is 0.926 / 0.917 / 0.825 / 0.908 (E1: 0.932 / 0.917 / 0.826 / 0.904).
- **H5.** Confident COP supercluster calls are at most 0.28% and confident broad OPC calls at most 2.3%. COP-derived calls make up 62–82% of confident OPC (plan: 62–86%).
- **H7.** Coverage clears its threshold by 0.08 to 0.22.
- **H8.** HD2_MERSCOPE is `broad_only` (A = 0.234). Its coverage of segmented objects, 0.152, is only 0.002 above the warning threshold (§7 item 3). Every other development sample is `full`.
- **H10.** Markers side with the new label in 96–99% of new-vs-legacy disputes. In WHB-vs-SEA-AD disputes the referee is close to a tie (E1: "roughly a tie").
- **OD-B13.** The below-60 SEA-AD rule costs 0.3–0.9 points of coverage. Replacing it with the likelihood typer would cost 2.0–5.8 points, so the LL variant keeps less coverage than the SEA-AD rule, not more. The trigger is not met.

Second vote on E2's 30k native cells (coverage of those cells; `second_vote_cost.csv`):

| Sample | Shared cells (share < 60 counts) | No second method | v1 without the below-60 rule | v1 (SEA-AD below 60) | v1 with LL below 60 | E2 LL rule alone | Below-60 SEA-AD cost | Below-60 LL cost | LL minus SEA-AD (trigger if > +0.08) |
|---|---|---|---|---|---|---|---|---|---|
| HD1_MERSCOPE | 30,000 (0.44) | 0.698 | 0.710 | 0.704 | 0.690 | 0.679 | 0.006 | 0.020 | -0.014 |
| HD1_XENIUM | 30,000 (0.61) | 0.597 | 0.606 | 0.603 | 0.551 | 0.543 | 0.003 | 0.054 | -0.052 |
| HD2_MERSCOPE | 30,000 (0.92) | 0.436 | 0.447 | 0.438 | 0.394 | 0.383 | 0.009 | 0.053 | -0.044 |
| HD2_XENIUM | 30,000 (0.83) | 0.500 | 0.508 | 0.505 | 0.451 | 0.443 | 0.003 | 0.058 | -0.055 |

M3 exit check against the E1 (iii) pilot (`exit_checks.csv`):

| Sample | Comparator | WHB supercluster, production bp >= 0.8 (n) [>= 0.98] | WHB supercluster, all shared cells | SEA-AD subclass, all shared cells [>= 0.925] | SEA-AD subclass, production bp >= 0.8 |
|---|---|---|---|---|---|
| HD1_MERSCOPE | E1 (iii) pilot | 1.0000 (16,644 of 30,000) | 0.9587 | 0.9573 | 0.9988 |
| HD1_XENIUM | E1 (iii) pilot | 1.0000 (13,625 of 30,000) | 0.9519 | 0.9453 | 0.9973 |
| HD2_MERSCOPE | E1 (iii) pilot | 1.0000 (9,722 of 30,000) | 0.9272 | 0.9483 | 0.9991 |
| HD2_XENIUM | E1 (iii) pilot | 1.0000 (11,275 of 30,000) | 0.9428 | 0.9413 | 0.9983 |

Both exit criteria hold on all four samples: the WHB supercluster agreement is 1.0000 at production bp >= 0.8, and the SEA-AD subclass agreement is 0.941–0.957 against 0.925. The third exit criterion, wall time within §10, is in §6.

## 4. Held-out donors: HD3 and HD4, proseg_hybrid (reported separately)

The flip rule scores H1–H3, H7, H8 and H12 on these datasets. Nothing in M3 was tuned on them, but E2, the evidence behind v1, used their data for these parameters and thresholds:

| Parameter or threshold | Derived with HD3 / HD4 data | Source |
|---|---|---|
| Raw thresholds (WHB broad / lineage 0.73, supercluster 0.69, SEA-AD broad 0.68) | Thinned high-count cells of HD3 M, HD3 X and HD4 X (`CALIB`, pooled with HD1 M / X and HD2 X) | `exp/E2/thresholds2.py` (`R2_pool = calib_real(T.ds.isin(CALIB))`), `exp/E2/common.py` |
| SEA-AD subclass thresholds (0.55 below 60 counts, 0.45 from 60) | The same `CALIB` pool | `exp/E2/thresholds2.py` R2 pooled |
| Packaged floors (`floors_human.csv`, broad and supercluster per class × platform) | The same `CALIB` datasets | `exp/E2/floors.py` (`T.ds.isin(CALIB)`) |
| H1 thresholds HD3 ≤ 0.17, HD4 ≤ 0.31 | Each donor's own E2 soft JSD (0.136 / 0.279) + 0.03 | Plan §14 H1 basis; E2 `cells_native` |
| H7 thresholds HD3 M ≥ 0.67, X ≥ 0.39; HD4 M ≥ 0.30, X ≥ 0.40 | Each dataset's own E2 LL-rule coverage − 0.10 | Plan §14 H7 basis; `exp/E2/out/final_coverage_and_gate.csv` |

So H1 and H7 on HD3 and HD4 test whether production reproduces the donors' own E2 measurement, not out-of-sample performance, and every rule on them runs with thresholds and floors partly fitted to their cells. What remains genuinely held out: the reseg segmentation (H17, §5; never used by E2) and, on the donors, H2, H3, H8 and H12, whose thresholds are generic (H2 1%, H3 0.80, H8 mechanical, H12 label-independent anatomy) rather than derived from the donors' values.

| Pair | H1 threshold | Soft, whole section [95% CI] | Soft, shared mask [95% CI] | Argmax, whole | Confident, whole | Soft, cells >= 30 counts | Set c soft / argmax | Unallocated soft mass M / X |
|---|---|---|---|---|---|---|---|---|
| HD3 | <= 0.17 | 0.126 [0.112, 0.143] | 0.107 [0.099, 0.117] | 0.110 [0.096, 0.126] | 0.156 [0.142, 0.167] | 0.326 | 0.110 / 0.096 | 0.016 / 0.017 |
| HD4 | <= 0.31 | 0.227 [0.212, 0.242] | 0.235 [0.218, 0.250] | 0.253 [0.237, 0.269] | 0.324 [0.307, 0.340] | 0.243 | 0.161 / 0.183 | 0.038 / 0.030 |

| Sample | Table cells | Median counts | A (share >= 30 counts) | Gate level | Warning | Confident broad, table cells [H7 threshold] | Confident broad, segmented objects | Lineage | Supercluster | Implausible calls (H2 <= 1%) | WHB-SEA 7-class, >= 20 counts [H3 threshold] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| HD3_MERSCOPE | 96,168 | 46 | 0.667 | full | no | 0.756 [>= 0.67] | 0.371 | 0.795 | 0.678 | 0.77% | 0.948 [>= 0.80] |
| HD3_XENIUM | 163,381 | 32 | 0.541 | full | no | 0.648 [>= 0.39] | 0.496 | 0.708 | 0.564 | 0.59% | 0.929 [>= 0.80] |
| HD4_MERSCOPE | 58,565 | 16 | 0.129 | broad_only | yes | 0.459 [>= 0.30] | 0.105 | 0.558 | 0.000 | 1.02% | 0.828 [>= 0.80] |
| HD4_XENIUM | 114,294 | 28 | 0.472 | full | no | 0.556 [>= 0.40] | 0.318 | 0.634 | 0.373 | 0.91% | 0.921 [>= 0.80] |

| Sample | COP argmax calls | Confident COP supercluster (H5 <= 2%) | Confident broad OPC (H5 <= 10%) | COP-derived share of confident OPC | COP suppressed (`flag_cop_suppressed`) | New vs legacy referee: disputes (share); new / legacy / tie (H10 >= 0.70) | WHB vs SEA-AD referee: disputes (share); WHB / SEA-AD / tie |
|---|---|---|---|---|---|---|---|
| HD3_MERSCOPE | 3.28% | 0.13% | 2.24% | 0.679 | 0.46% | 42,102 (0.438); 0.991 / 0.004 / 0.005 | 8,085 (0.084); 0.327 / 0.345 / 0.328 |
| HD3_XENIUM | 3.51% | 0.10% | 1.91% | 0.638 | 0.71% | 13,448 (0.082); 0.897 / 0.044 / 0.059 | 14,908 (0.091); 0.410 / 0.376 / 0.214 |
| HD4_MERSCOPE | 8.75% | 0.00% | 2.92% | 0.722 | 1.70% | 11,565 (0.197); 0.940 / 0.016 / 0.044 | 13,004 (0.222); 0.315 / 0.324 / 0.361 |
| HD4_XENIUM | 3.73% | 0.03% | 2.06% | 0.620 | 0.58% | 10,249 (0.090); 0.926 / 0.038 / 0.035 | 12,509 (0.109); 0.406 / 0.360 / 0.234 |

- **H1.** HD3 is 0.126 (CI 0.112–0.143; threshold 0.17) and HD4 is 0.227 (0.212–0.242; threshold 0.31). The plan's basis was 0.136 / 0.279 (the donors' own E2 values, table above). The pre-review independent-resampling upper bound for HD3 (0.171) crossed 0.17 purely as an artefact of mismatched anatomy.
- **H2.** HD3 is at 0.77% / 0.59%. **HD4_MERSCOPE is at 1.02%, above the 1% threshold** (§7 item 1); HD4_XENIUM is at 0.91%.
- **H3.** Agreement is 0.948 / 0.929 / 0.828 / 0.921, all at or above 0.80.
- **H7.** Every sample clears its threshold, by 0.09 to 0.26.
- **H8.** HD4_MERSCOPE is `broad_only` (A = 0.129) with the warning flag (coverage of segmented objects 0.105), as §5.4 expects. HD3 and HD4_XENIUM are `full`.
- **OD-B13.** The trigger is not met.

| Sample | Shared cells (share < 60 counts) | No second method | v1 without the below-60 rule | v1 (SEA-AD below 60) | v1 with LL below 60 | E2 LL rule alone | Below-60 SEA-AD cost | Below-60 LL cost | LL minus SEA-AD (trigger if > +0.08) |
|---|---|---|---|---|---|---|---|---|---|
| HD3_MERSCOPE | 30,000 (0.61) | 0.747 | 0.762 | 0.757 | 0.742 | 0.727 | 0.005 | 0.020 | -0.015 |
| HD3_XENIUM | 30,000 (0.71) | 0.637 | 0.648 | 0.646 | 0.572 | 0.561 | 0.002 | 0.076 | -0.073 |
| HD4_MERSCOPE | 30,000 (0.99) | 0.442 | 0.464 | 0.454 | 0.427 | 0.406 | 0.010 | 0.037 | -0.027 |
| HD4_XENIUM | 30,000 (0.77) | 0.543 | 0.556 | 0.551 | 0.535 | 0.523 | 0.004 | 0.021 | -0.016 |

The same MAP compared with E2's runs of the pilot configuration (for information; the formal exit check is §3):

| Sample | Comparator | WHB supercluster, production bp >= 0.8 (n) [>= 0.98] | WHB supercluster, all shared cells | SEA-AD subclass, all shared cells [>= 0.925] | SEA-AD subclass, production bp >= 0.8 |
|---|---|---|---|---|---|
| HD3_MERSCOPE | E2 run of the pilot configuration | 1.0000 (19,061 of 30,000) | 0.9701 | 0.9739 | 0.9995 |
| HD3_XENIUM | E2 run of the pilot configuration | 1.0000 (15,371 of 30,000) | 0.9573 | 0.9563 | 0.9988 |
| HD4_MERSCOPE | E2 run of the pilot configuration | 1.0000 (8,163 of 30,000) | 0.9987 | 0.9986 | 1.0000 |
| HD4_XENIUM | E2 run of the pilot configuration | 1.0000 (9,960 of 30,000) | 0.9397 | 0.9295 | 0.9968 |

## 5. Held-out segmentation: reseg (reported separately)

H17 is scored on HD1 and HD2 reseg: the H1 thresholds + 0.02, H2 and H9. The table shows H1 + 0.02 for every pair. HD3 and HD4 reseg are for information only. Reseg has no set-c run (the default `annotation_xplat_sensitivity_segmentations` is proseg_hybrid). HD2 and HD4 reseg used the declared panel (§2), and HD4 reseg MERSCOPE lacks 32 of the 297 panel genes (§7 item 4).

| Pair | H1 threshold | Soft, whole section [95% CI] | Soft, shared mask [95% CI] | Argmax, whole | Confident, whole | Soft, cells >= 30 counts | Set c soft / argmax | Unallocated soft mass M / X |
|---|---|---|---|---|---|---|---|---|
| HD1 | <= 0.19 | 0.127 [0.114, 0.142] | 0.125 [0.118, 0.132] | 0.129 [0.115, 0.143] | 0.128 [0.116, 0.143] | 0.259 | – | 0.008 / 0.009 |
| HD2 | <= 0.19 | 0.125 [0.110, 0.146] | 0.163 [0.152, 0.179] | 0.140 [0.113, 0.173] | 0.199 [0.182, 0.223] | 0.312 | – | 0.010 / 0.012 |
| HD3 | <= 0.19 | 0.177 [0.163, 0.191] | 0.160 [0.151, 0.170] | 0.177 [0.163, 0.192] | 0.188 [0.175, 0.203] | 0.444 | – | 0.007 / 0.011 |
| HD4 | <= 0.33 | 0.320 [0.303, 0.342] | 0.323 [0.301, 0.345] | 0.337 [0.319, 0.360] | 0.342 [0.324, 0.363] | 0.294 | – | 0.033 / 0.017 |

| Sample | Table cells | Median counts | A (share >= 30 counts) | Gate level | Warning | Confident broad, table cells [H7 threshold] | Confident broad, segmented objects | Lineage | Supercluster | Implausible calls (H2 <= 1%) | WHB-SEA 7-class, >= 20 counts [H3 threshold] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| HD1_MERSCOPE | 137,129 | 71 | 0.784 | full | no | 0.887 | 0.574 | 0.905 | 0.773 | 0.71% | 0.976 |
| HD1_XENIUM | 97,363 | 41 | 0.626 | full | no | 0.836 | 0.485 | 0.874 | 0.705 | 0.45% | 0.973 |
| HD2_MERSCOPE | 35,894 | 17 | 0.256 | broad_only | yes | 0.718 | 0.087 | 0.763 | 0.000 | 1.04% | 0.874 |
| HD2_XENIUM | 85,349 | 23 | 0.376 | full | no | 0.778 | 0.274 | 0.857 | 0.645 | 0.77% | 0.962 |
| HD3_MERSCOPE | 80,827 | 39 | 0.618 | full | no | 0.883 | 0.364 | 0.902 | 0.794 | 1.13% | 0.980 |
| HD3_XENIUM | 117,078 | 30 | 0.512 | full | no | 0.839 | 0.460 | 0.889 | 0.710 | 0.88% | 0.971 |
| HD4_MERSCOPE | 11,845 | 13 | 0.039 | broad_only | yes | 0.850 | 0.039 | 0.909 | 0.000 | 2.44% | 0.953 |
| HD4_XENIUM | 63,037 | 25 | 0.423 | full | no | 0.792 | 0.249 | 0.884 | 0.504 | 1.35% | 0.981 |

| Sample | COP argmax calls | Confident COP supercluster (H5 <= 2%) | Confident broad OPC (H5 <= 10%) | COP-derived share of confident OPC | COP suppressed (`flag_cop_suppressed`) | New vs legacy referee: disputes (share); new / legacy / tie (H10 >= 0.70) | WHB vs SEA-AD referee: disputes (share); WHB / SEA-AD / tie |
|---|---|---|---|---|---|---|---|
| HD1_MERSCOPE | 1.74% | 0.23% | 2.68% | 0.510 | 0.18% | 51,224 (0.374); 0.993 / 0.003 / 0.004 | 4,043 (0.029); 0.312 / 0.450 / 0.238 |
| HD1_XENIUM | 1.79% | 0.11% | 1.81% | 0.657 | 0.24% | 22,078 (0.227); 0.985 / 0.007 / 0.008 | 3,297 (0.034); 0.314 / 0.420 / 0.266 |
| HD2_MERSCOPE | 2.35% | 0.00% | 2.65% | 0.709 | 0.28% | 13,568 (0.378); 0.973 / 0.002 / 0.025 | 3,714 (0.103); 0.110 / 0.338 / 0.552 |
| HD2_XENIUM | 1.47% | 0.01% | 1.21% | 0.518 | 0.37% | 8,655 (0.101); 0.921 / 0.022 / 0.057 | 3,478 (0.041); 0.268 / 0.434 / 0.298 |
| HD3_MERSCOPE | 1.55% | 0.07% | 2.59% | 0.522 | 0.06% | 41,797 (0.517); 0.996 / 0.001 / 0.003 | 2,825 (0.035); 0.138 / 0.482 / 0.379 |
| HD3_XENIUM | 1.77% | 0.07% | 2.63% | 0.537 | 0.09% | 11,723 (0.100); 0.918 / 0.021 / 0.061 | 3,595 (0.031); 0.328 / 0.393 / 0.279 |
| HD4_MERSCOPE | 6.48% | 0.00% | 9.06% | 0.643 | 0.20% | 3,454 (0.292); 0.992 / 0.000 / 0.008 | 367 (0.031); 0.117 / 0.428 / 0.455 |
| HD4_XENIUM | 2.46% | 0.01% | 3.89% | 0.570 | 0.09% | 7,287 (0.116); 0.982 / 0.005 / 0.013 | 1,749 (0.028); 0.268 / 0.535 / 0.197 |

- **H17 / H1 + 0.02.** HD1 is 0.127 and HD2 is 0.125, both against 0.19. For information, HD3 is 0.177 (0.19 would apply) and HD4 is 0.320 (0.33).
- **H17 / H2.** HD1 is at 0.71% / 0.45%. **HD2_MERSCOPE reseg is at 1.04%, above 1%** (§7 item 1); HD2_XENIUM is at 0.77%. For information, HD3 MERSCOPE is at 1.13% and HD4 at 2.44% / 1.35%.
- **H17 / H9.** Not measured here (§8).
- Reseg has higher coverage and WHB–SEA agreement than proseg_hybrid on every sample (e.g. HD1_MERSCOPE broad coverage 0.887 vs 0.704, agreement 0.976 vs 0.926) over fewer table cells. This is input for E8 (OD-B6, OD-B7), not a criterion.

## 6. Resources (H14; §10)

| Pair | Segmentation | MAP wall (s) | MMC runs | WHB processor-s per 1k cells | SEA-AD processor-s per 1k cells | Max mapper peak RSS (GB) | Panel genes missing |
|---|---|---|---|---|---|---|---|
| HD1 | proseg_hybrid | 315 | 6 | 1.29–1.75 | 2.21–2.41 | 2.81 | 0 |
| HD2 | proseg_hybrid | 283 | 6 | 1.35–1.49 | 2.43–2.59 | 2.81 | 0 |
| HD3 | proseg_hybrid | 279 | 6 | 1.38–1.51 | 2.44–2.45 | 2.81 | 0 |
| HD4 | proseg_hybrid | 200 | 6 | 1.47–1.69 | 2.44–3.06 | 2.81 | 0 |
| HD1 | reseg | 179 | 4 | 1.32–1.49 | 2.26–2.54 | 2.81 | 0 |
| HD2 | reseg | 107 | 4 | 1.52–1.77 | 2.59–3.58 | 2.81 | MERSCOPE 1 |
| HD3 | reseg | 159 | 4 | 1.32–1.78 | 2.20–2.80 | 2.81 | 0 |
| HD4 | reseg | 80 | 4 | 1.59–3.72 | 2.82–7.50 | 2.81 | MERSCOPE 32 |

- MAP took 200–315 s per pair on proseg_hybrid, for 2 platforms × (WHB set a, WHB set c, SEA-AD). §10 budgets 25–40 min. Reseg (no set c) took 80–179 s. Both segmentations of a pair took at most 8.2 min, against H14's 2.5 h for all four segmentations.
- The mapper's peak RSS was at most 2.81 GB. The `merxen annotate` process peaked at 2.95 GB (`/usr/bin/time`, `$A/shadow/baselines/PROGRESS.txt`), against H14's 16 GB.
- WHB used 1.3–1.8 processor-seconds per 1k cells. §10 assumed 13.2–13.6; the host load was about 3–13. The 3.7 / 7.5 figures on HD4 reseg MERSCOPE (11,845 cells) are fixed start-up cost.
- No GPU was used.

## 7. Findings for the M3 PR and the gate PRs

1. **H2 fails at baseline on two scored datasets.** HD4_MERSCOPE proseg_hybrid is at 1.02% (held-out donor; the flip rule scores H2 on HD4), and HD2_MERSCOPE reseg is at 1.04% (H17 includes H2). For information, HD3_MERSCOPE reseg is at 1.13% and HD4 reseg at 2.44% / 1.35%. The calls are mostly Amygdala excitatory (region-implausible for frontal cortex in the vocab) and Miscellaneous.
   - Counting only implausible calls whose lineage probability passes 0.73 would give 0.57% and 0.93%. That changes the metric definition in the passing direction, so it is a loosening and needs the user's written approval (§1 rule 2).
   - Nothing is changed here. The gate PR needs a fix, for example in RESOLVE's status precedence (M4) or the vocab's plausibility of Amygdala excitatory, or a written, user-approved exception.
2. **OD-B13 is not triggered.** On all eight datasets the below-60 SEA-AD rule removes 0.2–1.0 points of confident broad coverage (v1 without the rule minus v1). Using the likelihood typer as the below-60 vote instead would remove 1.4–7.3 points more (`ll_minus_sea` −0.014 to −0.073). E2's LL rule alone, with no SEA-AD role, keeps 2.5–8.5 points less than v1. The LL part of M10 therefore stays in v1.1, to be recorded in the M3 PR. The LL value test (M3 item 6; the LL (vii) recipe on the WHB-frontal bundle profiles) confirms this on every table cell (§11.5).
3. **H8 margin.** HD2_MERSCOPE's coverage of segmented objects (0.152) is 0.002 above the 0.15 warning threshold, and §5.4 expects no warning there. RESOLVE's coverage may move by up to ±0.05 (M4 exit), so HD2_MERSCOPE may acquire the warning flag. The warning never changes the gate level.
4. **HD4 reseg MERSCOPE lacks 10.8% of the panel.** The published `var` of 11,845 cells is `min_cells`-filtered to 268 features. M3 maps the present genes with a restricted lookup; plan §3.3 / §8.1 route more than 5% missing genes to a panel family of its own with a full PREP (M3b). Its reseg baselines are provisional until M3b re-measures them. HD2 reseg MERSCOPE lacks 1 gene (0.34%), within the 1% that a restricted lookup covers.
5. **Standalone runs on `min_cells`-filtered outputs.** `merxen annotate --from-clustered-h5ad` derives the panel from the published `var`. When a small sample lost a gene, the derived panel gets a new hash and the store has no bundle for it (HD2 and HD4 reseg failed this way). The declared panel has to be supplied through `merxen annotation-panel --panel-file` and `--panel-dir`. Adding `--panel-file` to `merxen annotate` is an open follow-up.
6. **Candidate tightenings** (not applied; for decision in the M3 PR):
   - **H1** (basis "soft + 0.03", now measured on production): HD1 ≤ 0.16, HD3 ≤ 0.16, HD4 ≤ 0.26 (HD2 stays at 0.17).
   - **H3:** HD3 ≥ 0.90 (baseline 0.948 / 0.929).
   - **H5:** confident COP supercluster ≤ 1% (maximum baseline 0.28%) and confident broad OPC ≤ 5% (2.9%).
   - **H7** (basis "E2 LL-rule coverage − 0.10"; now the SEA-rule baseline − 0.10 where that is stricter): HD1 X ≥ 0.50, HD2 X ≥ 0.41, HD3 X ≥ 0.55, HD4 M ≥ 0.36, HD4 X ≥ 0.46.
   - **H10:** ≥ 0.85 (minimum baseline on the flip-rule datasets 0.961).

   Resolvability (M3b) and RESOLVE (M4) can still lower coverage, so tightening H7 before M4 risks a failure that belongs to M3b.
7. **H4's "assigned class" is an open choice for the user** (§11.3). Under the M3 working metric (a), the re-map's argmax over all table cells, H4 fails at baseline on HD2 (2/7 and 5/7 on MERSCOPE / Xenium) and HD3_MERSCOPE (5/7), against ≥ 6 of 7; fold ≥ 3 holds everywhere and the AUROC fails, for OPC on every sample, because the argmax OPC class holds the COP calls the v1 COP rule never emits as OPC. With the COP rule applied (b) only HD2 fails (2/7, 5/7); with its SEA-AD rescue (b′) or on WHB-only confident calls (c) only HD2_MERSCOPE fails (2/7 and 4/7). The user picks the definition in the M3 PR; a HD2_MERSCOPE failure under any option needs a fix or a written, user-approved exception at gate H.
8. **OD-B8 goes to the user with both readings** (§11.5): as the v1.1 below-60 vote LL cannot pass by construction (coverage gain ≤ the below-60 SEA-AD cost, at most 1.0 point); as a tie-breaker it moves referee outcomes by +6 to +14 points, a circular measure. The typer is the LL (vii) recipe on the WHB-frontal bundle profiles; scoring (vii) proper needs `ll_whb_ctx_profiles` built first.
9. **The provisional labels of the shadow MAP runs predate the review fixes.** `<sid>_ct_provisional.parquet` and `provisional_summary` in `$A/shadow/{baselines,e8,stageA}/runs` were written before the float32 tolerance and the SEA-AD subclass `aggregate_probability`; their `ct_*_status` can differ at cells exactly at a threshold (0.7–1.3% of table cells at supercluster bp 0.69, about 1% at WMB class bp 0.90). No number in this document reads those statuses: every rule above is re-evaluated from the raw engine columns by the shadow scripts.

## 8. Not measured in this stage

| Criterion | Where it is measured |
|---|---|
| H6 simulation precision of the v1 thresholds | the archived E2 HO simulation, scored at M8; H18 recomputes it in production |
| H9 canonical-marker plausibility (also part of H17) | M8 acceptance script |
| H12 cortical depth, H13 contracts, H15 reproducibility | M5 / M7 / M8 |
| H16 realised flag rates (production flags) | M4 (the M3 prototype is §11.4) |
| H18 resolvability regression | M3b (production PREP) |
| MO1–MO11 (mouse) | M6, M6b, M9 |
| NP1–NP9 (gate P) | M13 |

M3 shadow items 2–7 (E8 segmentation comparison, X1 rescaling, held-out genes, flag rates, the LL value test and the glial JSD difference) are reported in §11.

## 9. Registered thresholds (plan rev3 §14, verbatim)

The text below is §14 of `docs/plans/robust-celltype-annotation-plan.md` as of `06a04a1`. Section references (§n) point into the plan. The section title is shown in bold rather than as a heading; the text is otherwise unchanged. The user's decisions of 2026-09-27 on the tested-set rule (one rule for §8.3 resolvability and gate P) and on the human held-out test set are recorded in §12; they do not change a threshold.

**14. Acceptance criteria (pre-registered; must pass before each gate)**

**Rules.** Thresholds are fixed **now** and copied into `docs/acceptance/annotation-v1-preregistration.md` at M3 (the MO10 rule variant at M6b). Baselines may only **tighten** a threshold; loosening needs your written approval in the gate PR (R-sci). **Held out:** HD3 and HD4 (donors), reseg (segmentation, HD1 / HD2), the MERFISH AP series (MO10; region shares recomputed leaving each section and its neighbours out) and, for gate P, the simulation replicates on which frozen thresholds are tested (other held-out donors or test draws and seeds, NP4); none is used for tuning beyond the pre-registered selection rule, and their results are reported separately. Metrics come from `acceptance_metrics.json` and the acceptance scripts, on proseg_hybrid unless stated; [L] = first-run target.

### Human (gate H, at M8)

| ID | Criterion | Threshold | Basis |
|---|---|---|---|
| H1 | Soft broad JSD (7 classes, all table cells, whole section; 95% block-bootstrap CI and shared-mask value reported) | HD1 ≤ 0.17; HD2 ≤ 0.17; **HD3 ≤ 0.17; HD4 ≤ 0.31** (held out) | Soft .139 / .129 / .136 / .279 + 0.03 (`review_scientific/confident_jsd_bias_extra.txt`, E2 `cells_native`); E1 `real_jsd.csv` (iii) .132 / .143 under its scheme, so HD2 gets the larger margin [M]; legacy .51 / 1.00 / .66 / .44 |
| H2 | `flag_implausible` share of table cells | ≤ 1% per dataset | E1 (iii) 0.3–0.6% |
| H3 | WHB–SEA 7-class agreement, table cells ≥ 20 counts | HD1 M / X ≥ 0.90; HD2 M / X ≥ 0.80; HD3, HD4 ≥ 0.80 | E1 .932 / .917 / .826 / .904 |
| H4 | Held-out-gene enrichment (§5.8), per platform | Fold ≥ 3 and AUROC ≥ 0.70 in ≥ 6 of 7 broad classes on HD1, HD2, HD3 [L] | Non-circular (R-sci) |
| H5 | COP control | Confident COP supercluster ≤ 2%; confident broad OPC ≤ 10%; COP-derived OPC share reported | E1; `review_scientific/cop_demotion_rule.csv` |
| H6 | Simulation precision of the v1 raw thresholds (archived E2 HO contaminated simulation, reweighted to each dataset's composition) | Broad ≥ 0.90 at D ≥ 15; supercluster ≥ 0.85 at D ≥ 30, classes with n ≥ 50 [M] | Replaces the draft's circular curve test; H18 recomputes it in production |
| H7 | Confident broad coverage of table cells | HD1 M ≥ .62, X ≥ .44; HD3 M ≥ .67, X ≥ .39; HD2 M ≥ .34, X ≥ .29; HD4 M ≥ .30, X ≥ .40 | E2 LL-rule coverage − 0.10 (SEA-rule cost unmeasured until M3) |
| H8 | Gate levels | HD2_M and HD4_M `broad_only` (HD4_M with the warning flag); none `failed`; the rest `full` (mechanical, given A). Real-data QC (§8.8) only warns on these seeded datasets before this gate; its expected warnings are listed in the gate PR (e.g. HD4's paired broad JSD, soft .279 > 0.20; the marker-referee thresholds were measured only on HD1 and HD2, E1) | §5.4, §8.8 |
| H9 | Canonical-marker plausibility of confident `ct_broad` (marker pseudo-labels, `data/HD1` §C, `data/HD2` §4) | HD1 M ≥ 0.90, X ≥ 0.83; HD2 M / X ≥ 0.75 [M] | Legacy 69.6% / 44.2%; MMC at bp ≥ 0.9 93% / 85% |
| H10 | Marker referee on new-vs-legacy broad disputes | Markers side with the new label in ≥ 70% | E1 referee method |
| H12 | Cortical depth from manual pia/WM boundaries (label-independent) | HD1, HD3: Upper-layer IT < Deep-layer IT < Deep-layer NP/CT/6b medians with non-overlapping 95% CIs on both platforms. **All 4 pairs incl. HD2, HD4, at broad level:** oligodendrocytes WM > GM on both platforms [L] | R-sci; outputs exist for all 4 pairs × 2 platforms [H] |
| H13 | Contracts | MENDER completes on all 4 pairs under both unassigned policies; cortical-depth violins; H5AD and zarr id sets identical; legacy tables untouched before the flip (sha256); viewer on reseg (manual) | Wiring map §c |
| H14 | Resources | Annotation wall ≤ 2.5 h per pair (all segmentations); MAP peak RSS ≤ 16 GB; PREP ≤ 2.5 h per validated panel incl. resolvability; no GPU | §10 |
| H15 | Reproducibility | Identical re-run → identical DataFrame content; seed 0 vs 1 changes ≤ 1% of confident broad labels | E5: 99.85% stable at bp ≥ 0.8 |
| H16 | Flag transparency | Realised rates for every flag × class × platform; strata > 15% marked uninformative (mechanical) | §5.6 |
| H17 | Segmentation hold-out (reseg, HD1 / HD2) | H1 thresholds + 0.02; H2; H9 | R-sci |
| H18 | (rev2) Resolvability regression on the validated panels, from production PREP | Human set a (post-M0e): broad and supercluster emitted for every class with ≥ 50 test cells at depths from its v1 floor plus one grid step up to D_max; WHB cluster not emitted; simulated precision at 0.73 / 0.69 ≥ 0.90 / 0.85 at D ≥ 15 / 30 (local rule). MS1, MS2: class and subclass emitted at min(dataset median depth, D_max(c)) for every class with ≥ 50 test cells. All three `validated`. No validated threshold would be raised, or each exception is listed in the gate PR | §8.3; E2; mouse report §4.2 |

**Human flip rule:** H1–H3, H5, H7–H10, H12–H16 on HD1 and HD2; H1–H3, H7, H8, H12 on HD3 and HD4; H4 on HD1, HD2, HD3; H6 and H18 once; H17 on reseg. Any failure needs a fix or a written, user-approved exception in the gate PR. The draft's split-rule criterion (H11) belongs to M11.

### Mouse (gate M, at M9)

| ID | Criterion | Threshold | Basis |
|---|---|---|---|
| MO1 | Class marker consistency over **all marker-pseudo-confident cells** (fixed denominator; confident-cell value reported) | MS1 proseg_hybrid ≥ 0.89; MS2 original_seg ≥ 0.84 | .911 (`review_feasibility/mo1_ms1_marker_consistency.csv`), .856 (`review_scientific/mouse_ms2_marker_consistency_panelLK.csv`) |
| MO2 | Pruning mechanics (not validation) | T2 after pruning ≤ 0.1%; class changes outside dropped nodes ≤ 0.25%; subclass ≤ 1.5% | E7 |
| MO3 | Inferred regions | = manual sets for MS1 and MS2 | E7 §4 |
| MO4 | Soft class composition vs AP-matched MERFISH | MS2 original_seg Immune ±0.5 and Astro-Epen ±3 points (MERFISH .31–.33: 1.50–1.56 / 17.2–17.5); MS1 ±1 / ±5 [L; age caveat] | Replaces the draft's circular MO4 / MO6 (R-sci) |
| MO5 | Spill-over flag | MS2 held-out microglial-gene enrichment ≥ 10×; astrocyte FPR ≤ 0.5%; never shown as a count | E3 (35–40×) |
| MO6 | F1 rate after pruning | 0.2–1.0% | E7 |
| MO7 | Mouse gate | Invalid MS2 proseg_hybrid `failed`; MS2 original_seg and MS1 not `failed` | §7.6 |
| MO8 | Resources | ≤ 1.25 h per 4-segmentation section; PREP ≤ 2 h per panel incl. resolvability; MAP peak ≤ 12 GB; PREP peak ≤ 40 GB at 8 processes (query markers measured 20–27 GB at 16) | §10, §8.7 |
| MO9 | MS2 after the M0a re-run | Registration passes; reseg / proseg_hybrid soft class JSD vs original_seg ≤ 0.05 [L] | Gates the *use* of those outputs, not the flip |
| MO10 | (rev2) AP-axis harm on MERFISH-638850 with the production rule (M6b), per AP bin over the 38 in-scope sections, for each of 3 gene sets, with region shares computed leaving each section and its neighbours out. Harm = share of confident Allen-labelled cells (`average_correlation_score ≥ 0.6` and our class bp ≥ 0.9) whose class or subclass is in the production drop list | Per bin mean harm ≤ 0.10% and per-section maximum ≤ 0.30%; T2 after pruning ≤ 0.1% in every bin; missed regions reported and allowed only where that section's harm ≤ 0.1%; oracle-relative harm reported as a diagnostic [L] | §7.8 (direct Allen-label proxy today: AP 3.0–4.5 mean / max 0.15% / 0.70%, every other bin ≤ 0.08% / 0.11%) |
| MO11 | (rev2) First real anterior coronal section (AP < 6.0 mm) | Inferred regions = manual anatomy, or differ only by slivers < 1% of tiles with harm ≤ 0.1%; MO2 mechanics; gate not `failed`; G4 within warning bands or explained [L] | §7.8 step 3 |

**Mouse flip rule:** MO1–MO8 and MO10. MO11 is required before map_first results from anterior sections (AP < 6.0 mm) lose their banner; it does not block the flip.

### New panel family (gate P; before a family is listed as `validated`)

**Simulation only** (OD-E1 / OD-E9, decided 2026-09-26). Scored by `scripts/acceptance/new_panel.py` (`annotation-panel-simulate --gate-p`) on the family's declared panel: held-out Allen reference cells restricted to the panel, thinned to the grid (and weighted to a family dataset's depth histogram when one exists, label-free), perturbed and mapped with the production configuration (§8.3, §8.8). No real dataset is needed or used for promotion. Notation: target_L = 0.90 (broad, lineage, NT, class) or 0.85 (supercluster, subclass); **target⁺_L** = target_L + 0.05 at ≥ 60 counts and + 0.10 below 60 counts, capped at 0.97 (the provisional targets, §8.2). These margins are stricter than for the real-data-validated panels (H6: point precision ≥ target_L; the §8.3 emission rule: Wilson bound ≥ target − 0.02 and coverage ≥ 0.2), because simulation is optimistic (E2 verdict 5: native low-count cells carry 30–45% more genes than thinned cells; up to 5–10 points below ~60 counts on Xenium). Replicates, stress recipes and the evaluation constants: §3.7 (`gate_p_*`).

**Evaluation rules** (rev3 check; pre-registered with the thresholds). They keep the criteria passable at the sample sizes the references supply. A Wilson bound ≥ 0.95 needs ≥ 73 confident calls even at 100% precision, and the three frontal WHB donors together hold few glia on set a: Microglia 1,198 cells (483 with ≥ 60 panel counts, 28 with ≥ 120), OPC 1,833, Vascular 37, Fibroblast 25; the default HO donor alone has 243 Microglia (R45) [H].
- **Pooled held-out calls.** NP3, NP6 and NP7 are scored at the frozen thresholds on all held-out calls: human, the default donor's check half (split halves, §8.3) plus both other donors, each mapped against the bundle built without it; mouse, both draws. NP4 then compares the replicates.
- **Tested sets.** Per (level L, class c), a depth bin is tested on its own only if it holds ≥ n_min = 200 confident calls (`gate_p_min_confident_n`). From the deep end, bins are pooled cumulatively into a "≥ d" set, each test cell counted once at its deepest bin, until the set holds ≥ n_min calls. The set's shallowest bin is **D_P(L, c)**; its result applies to every depth ≥ D_P, and bins deeper than D_P are marked extrapolated (as §8.3 does beyond D_max). A (level, class) with < n_min confident calls in total is `not_evaluable`. Wilson bounds on reweighted sets use the Kish effective n.
- **Class set C_P.** The classes with ≥ 700 pooled test cells (`gate_p_class_min_test_cells`, ≈ n_min / 0.30), fixed from the test-cell table before any cell is mapped. The second mouse draw tops every class up to 700 where the 10Xv3 cells unused by the marker build allow; it takes ≤ 5% of any cluster's cells, which bounds leakage into the Allen means [L]. C_P must hold ≥ 90% of the pooled test cells, and the PR lists the excluded classes with their share of the reference composition.
- **Per-class records.** A (level, class) is validated when NP3–NP7 pass for it. `validated_min_depth(L, c)` is the shallowest bin from which every bin below D_P is tested on its own and passes NP3, and the ≥ D_P set passes too. The record (§4.7) carries it, with D_P as `tested_max_depth`, or else the failing criterion. A failing or unevaluable class stays provisional per class (§8.2) and does not block the others.
- **Expected depth** (NP5): the median native panel counts of a family dataset when one exists, else the vendor's or the panel design's median depth, stated in the PR.
- **Dry run.** Before any new family is scored, the programme runs on the seeded families (set a, MS1, MS2). They must pass at broad (human) / class (mouse) for every C_P class, and at supercluster / subclass for the classes H18 expects. If they do not, the criterion (statistic, n_min or margin) is unattainable rather than the panel bad; it is revised in a PR you approve (loosening rule, §14 Rules), and the revision is recorded before any new family is scored.

| ID | Criterion | Threshold | Basis |
|---|---|---|---|
| NP1 | Gene IDs and controls (declared panel file) | ≥ 95% of non-control features resolved to the run's species (≥ 98% when the vendor supplies Ensembl IDs); the exact-case species test passes; every control probe / codeword type named in the 10x or Vizgen documentation is removed by the registry and none reaches the mapping query; the unresolved list reviewed | §8.4 |
| NP2 | Panel coverage | Root markers ≥ 10; every root child with n ≥ 50 has ≥ 10 markers; weak and collapsed parents accepted in the PR | §8.2 |
| NP3 | Precision and coverage at the panel's depth (base recipe R1_contam_HO, seed 0, pooled held-out calls at the frozen thresholds) | For each level up to `max_leaf_level` and each class with ≥ n_min confident calls, at every tested set from `validated_min_depth(L, c)` up to and including the ≥ D_P set: point precision ≥ target⁺_L (the margin of the set's shallowest bin), Wilson 95% lower bound ≥ target_L and coverage ≥ 0.30, reweighted both to the reference's natural composition within the class and to a class-balanced one; values on a family dataset's depth histogram and composition reported when one exists [L] | Stricter than H6 and the §8.3 emission rule. At n = 200 the bound (0.910 at 0.95, 0.851 at 0.90) binds only when reweighting shrinks the effective n |
| NP4 | Stability across held-out donors and seeds (frozen thresholds) | Replicates: human, leave-one-donor-out over H19.30.002, H19.30.001 and H18.30.002 (each with its own HO bundle) × seeds 0 / 1; mouse, the two disjoint test draws × seeds 0 / 1 (the Allen means contain every donor, so a mouse donor cannot be held out; the per-donor spread is reported from the ABC `donor_label` [L]). Per (level, class), at every tested set: each replicate with ≥ 100 confident calls there (`gate_p_replicate_min_confident_n`) has point precision ≥ target_L. Where every replicate has ≥ 100 calls, the range of the seed-averaged donor (or draw) precisions is ≤ max(0.03, 3.5 × pooled SE), with pooled SE = √(p̄(1 − p̄)/n̄); by sampling alone, three equally precise replicates exceed 2 × SE 33% of the time, but 3.5 × SE only 3.5% (two replicates: 1.3%). Seed 0 vs 1 changes ≤ 2% of confident labels per validated level [L] | Seed noise 5.5% of SEA-AD subclass (E5), 1.6–3.0% of WMB subclass labels (E7); bp ≥ 0.8 cells 99.85% stable |
| NP5 | Resolvability consistency | Per (level, class), the §8.3 emission decisions re-derived in each replicate agree with the base run at every depth bin with ≥ 50 test cells, except at most the bin adjacent to the emission boundary; each t* varies ≤ 0.05 across replicates at tested sets; a class is not validated at L if its cells at the family's expected depth would be > 50% `resolvability_extrapolated` [L] | §8.3 (D_max, extrapolation) |
| NP6 | Sensitivity to contamination and gene-efficiency perturbations | With the frozen thresholds, pooled over every held-out donor or draw at seed 0: spill 0.35 (base 0.25); gene efficiency LogNormal(0, 1.0) (base 0.8); per-gene log2 factors drawn from the measured human cross-platform offsets (capped ±2). At NP3's tested sets (those left with < n_min stressed calls are pooled with the next deeper set): point precision ≥ target_L and Wilson lower bound ≥ target_L − 0.02 (the real-data emission rule, no margin), and a drop in point precision vs the base recipe not significantly above 0.05 (one-sided 95%); the clean (thinning-only) upper bound and coverage changes reported [L] | E2 R1_contam_HO; per-pair offsets −0.12 to +1.52 log2 (D-A7) |
| NP7 | Error structure (in-silico analogue of H2 / H5) | Human: confident calls to sink or region-implausible nodes ≤ 1% of confident calls. Human and mouse: at every tested set of a validated (level, class), no single wrong node receives > 5% of the class's confident calls (e.g. COP-like over-calling) [L] | H2, H5, D-A6 |
| NP8 | Cross-panel support (only when the family will be paired with a different panel) | The intersection panel with each intended partner meets NP3 at broad level; otherwise cross-platform statistics for such pairs stay broad-level and flagged (§8.5). Real paired concordance is production QC (downgrade-only, §8.8) [L] | §8.5 |
| NP9 | Resources, reproducibility | PREP incl. resolvability and the gate-P replicates within 1.5× of §8.7 / §10 (for 5K, of the M3b measurement); peak RSS of every PREP step, query markers included, within the reserve; above 1,000 genes the per-parent prefilter agrees with the unfiltered lookup ≥ 0.95 per validated level and class with n ≥ 50 at bp ≥ 0.8, with no parent below 5 markers; identical re-run → identical bundle and table content | H14 / H15; §8.7 |

**Gate-P rule:** NP1, NP2 and NP9 for the family (NP8 when it will be paired with a different panel) and NP3–NP7 per (level, class), all from simulation and scored under the evaluation rules above. `validated_max_level` is the deepest level ≤ `max_leaf_level` at which every class of C_P is validated. It is the family's headline level, and promotion requires at least broad (human) / class (mouse). The `validated_panels.csv` row carries `validation_basis = simulation`; the `validated_panel_levels.csv` rows carry each (level, class)'s status, `validated_min_depth` and `tested_max_depth` (§4.7). The family keeps the provisional emission rule and margins in production, so promotion never changes what is emitted (§8.2). Promotion by a PR you approve (into the integration branch before the first species gate, to `main` after, §2.2). Real datasets never promote: the production QC (§8.8) can only warn, force `broad_only` or `not_resolvable` for a dataset, or trigger a demotion PR.

## 10. File index

All paths are under `$A/shadow/baselines/`.

| File | Content |
|---|---|
| `metrics/sample_metrics.csv` | Per sample × segmentation: gate (A, level, warning, reasons), coverage per second-vote variant (lineage / broad / supercluster, table and segmented), implausible share and top nodes, COP shares, WHB–SEA agreement (E1 and strict) |
| `metrics/jsd.csv` | Per pair × segmentation × kind (soft, argmax, confident, soft_ge30, setc_soft, setc_argmax) × region (whole_section, shared_mask): JSD, 95% CI over matched tiles (`resampling` = joint), the independent-resampling sensitivity (`ci_*_independent`), tile locations, tiles per section, cells, unallocated mass |
| `metrics/compositions.csv` | The compositions behind `jsd.csv`, per platform (shares incl. unallocated, and renormalised 7-class shares) |
| `metrics/referee.csv` | E1 marker referee: WHB vs SEA-AD, new (v1 shadow) vs legacy, WHB argmax vs legacy, new vs SEA-AD |
| `metrics/second_vote_cost.csv` | OD-B13: coverage under the five second-vote variants on E2's 30k native cells, costs, trigger, E2's published LL-rule coverage |
| `metrics/exit_checks.csv` | M3 exit: production vs pilot agreement per sample, reference and subset |
| `metrics/runtime.csv` | MAP wall time, processor-seconds per 1k cells, peak RSS, missing panel genes per run |
| `runs/<pair>/<seg>/` | `merxen annotate` outputs: `map_manifest.json`, `panel/`, `<plat>/<sid>_mmc_<run>.parquet`, `<plat>/<sid>_ct_provisional.parquet`, `logs/`; for HD2 / HD4 reseg also `panel_varderived/` (the refused var-derived panel) and `inputs_view/` |
| `logs/`, `PROGRESS.txt` | Run logs with `/usr/bin/time` wall and max RSS; `metrics.log` |

## 11. Shadow programme items 2–7 (M3 stage C2)

Measured 2026-09-27 with the same MAP configuration as §2 (no engine change); re-measured after the M3 review with matched-tile CIs, E2's SEA-AD definitions and the float32 threshold tolerance (header). Reports: `$A/shadow/<item>/REPORT.txt`; summary: `$A/shadow/SHADOW_SUMMARY.txt`. Code: `merxen.annotation.likelihood` (LL (vii)), `merxen.annotation.shadow` (additions below), `scripts/acceptance/{shadow_e8,shadow_x1,heldout_genes,shadow_flags,shadow_ll,shadow_glial_jsd}.py`. HD3, HD4 and reseg stay held out: they are reported, and nothing below was selected on them.

### 11.1 E8 segmentation comparison (OD-B6, OD-B7)

HD2 on all four segmentations (both platforms) and MS1 on all four (MERSCOPE, WMB MS1 bundle `d2e3f496`). Confident cells per mm² use the median tissue area of the section's segmentations (100 µm bins with ≥ 3 table cells). Foreign fraction = counts on the assigned class's negative genes / total counts. Mouse class confidence counts a class bp of exactly 0.90 (float32 tolerance, §2), which adds about 1% of cells to every segmentation.

| Sample | Metric | original_seg | proseg_mask | proseg_hybrid | reseg |
|---|---|---|---|---|---|
| HD2_MERSCOPE | Confident broad per mm² (ratio vs proseg_hybrid) | 654 (1.06) | 609 (0.99) | 616 (1.00) | 352 (0.57) |
| HD2_XENIUM | Confident broad per mm² (ratio) | 1,048 (0.95) | 1,091 (0.99) | 1,099 (1.00) | 908 (0.83) |
| HD2 | Soft broad JSD, MERSCOPE vs Xenium [95% CI, matched tiles] | 0.153 [0.146, 0.161] | 0.130 [0.123, 0.140] | 0.133 [0.126, 0.142] | 0.125 [0.110, 0.146] |
| HD2_MERSCOPE / _XENIUM | Foreign (negative-gene) fraction of confident cells | 0.022 / 0.022 | 0.019 / 0.022 | 0.019 / 0.021 | 0.005 / 0.012 |
| MS1 | Confident class / subclass per mm² (ratio) | 2,094 / 1,677 (1.11 / 1.05) | 1,877 / 1,584 (1.00 / 0.99) | 1,883 / 1,593 (1.00 / 1.00) | 2,086 / 1,786 (1.11 / 1.12) |
| MS1 | Foreign fraction; confident microglia | 0.0137; 136 | 0.0132; 255 | 0.0128; 294 | 0.0100; 1,114 |

- **OD-B6: the switch criterion is not met.** No segmentation has ≥ 25% more confident broad cells per mm² on HD2_MERSCOPE (best: original_seg, +6%), and all four leave it `broad_only` (A 0.21–0.26). The adopted policy (proseg_hybrid, broad level only) stands.
- **OD-B7, recommendation for the user:** per dataset. Human: keep proseg_hybrid (proseg_mask is equivalent; reseg has purer cells and a lower platform JSD but 17–43% fewer confident cells per mm²; original_seg has the highest platform JSD). Mouse MS1: reseg (most confident subclass cells, class cells equal to original_seg, lowest foreign fraction, almost four times the confident microglia). MS2 is re-checked after the M0a re-run.

### 11.2 X1 platform rescaling (OD-B9)

Per-gene factors against the reference pseudobulk of each dataset's cells with WHB bp ≥ 0.8, median-centred and capped at ±2 log2, then a WHB set-a re-map. The C2 run (2026-09-27 01:06–01:12) scored X1 on a pass rule with a referee condition; that rule was written into `scripts/acceptance/shadow_x1.py` but committed after the run, and the logs recorded no script hash, so it cannot be shown to have been fixed before the runs. It is not used here. The decision below rests on effect size; the review rerun's launcher log records the script's sha256.

| Pair | Soft JSD set a / set c / X1 | X1 − set a [paired 95% CI] | Set c − set a [paired 95% CI] | Referee X1-vs-set-a disputes, X1 / set a wins (MERSCOPE; Xenium) | Referee set-c-vs-set-a disputes, set c / set a wins (MERSCOPE; Xenium) |
|---|---|---|---|---|---|
| HD1 | 0.129 / 0.121 / 0.128 | −0.001 [−0.002, −0.001] | −0.008 [−0.010, −0.006] | 0.37 / 0.40; 0.24 / 0.63 | 0.30 / 0.49; 0.23 / 0.65 |
| HD2 | 0.133 / 0.109 / 0.126 | −0.007 [−0.008, −0.006] | −0.024 [−0.035, −0.016] | 0.25 / 0.40; 0.21 / 0.58 | 0.20 / 0.53; 0.23 / 0.57 |
| HD3 (held out) | 0.126 / 0.110 / 0.126 | 0.000 [−0.000, 0.001] | −0.016 [−0.018, −0.014] | 0.33 / 0.38; 0.26 / 0.54 | 0.28 / 0.49; 0.26 / 0.58 |
| HD4 (held out) | 0.227 / 0.161 / 0.217 | −0.010 [−0.010, −0.009] | −0.066 [−0.069, −0.062] | 0.37 / 0.33; 0.28 / 0.53 | 0.28 / 0.44; 0.26 / 0.55 |

(Paired CIs over matched tiles, §2; the independent-resampling sensitivity is in `x1_jsd_differences.csv`.)

**Decision: X1 is not adopted** (OD-B9). On the development pairs its soft-JSD gain over set a is at most 0.007 (0.001 and 0.007), 18% and 29% of set c's gain (0.008 and 0.024), and H4 is unchanged (§11.3). The E1 referee is **not neutral** for rescaled labellings: it scores raw panel counts, so it favours labellings mapped from unrescaled counts. It sides with set a against set c as well (set c wins 0.20–0.30 of their disputes, set a 0.44–0.65), yet set c is the mandatory sensitivity, so the referee outcome is reported but not used as a criterion. `annotation_xplat_sensitivity` stays `geneset_c`; plan §17 keeps per-platform rescaling as a v2 candidate. The factors have a spread of 2.4–3.0 log2 and 36–50% of genes hit the cap, because they mostly measure reference (snRNA-seq) vs in-situ efficiency; their Xenium-minus-MERSCOPE difference correlates 0.80–0.92 with the measured cross-platform ratios (`research/xplat`).

### 11.3 H4 held-out-gene baseline

Metric as §2, with the "assigned class" left to the user (options (a)–(c) below). Markers on set a: Neurons SLC17A7, GAD1 and SLC17A6 (MERSCOPE) or GAD2 (Xenium); Astrocytes AQP4, GJA1; Oligodendrocytes MOBP, MOG, OPALIN; OPC PDGFRA, VCAN; Microglia CX3CR1, GPR34 (+ P2RY12 on Xenium); Vascular FLT1, PECAM1, ABCC9; Fibroblasts DCN, FBLN1, COL12A1 (`$A/shadow/heldout/heldout_markers.csv`). The asset's rank-2 alternates were added in M3 because set a lacks SATB2, PLP1, CSF1R, CLDN5 and LUM.

| Sample | (a) argmax, all table cells (M3 working metric) | (b) argmax with the COP rule | (b′) (b) with the SEA-AD OPC rescue | (c) WHB-only confident | Classes failing under (a) |
|---|---|---|---|---|---|
| HD1_MERSCOPE | 6 | 7 | 7 | 7 | OPC |
| HD1_XENIUM | 6 | 6 | 7 | 7 | OPC |
| HD2_MERSCOPE | 2 | 2 | 2 | 4 | Neurons, Oligodendrocytes, OPC, Microglia, Vascular |
| HD2_XENIUM | 5 | 5 | 6 | 7 | OPC, Microglia |
| HD3_MERSCOPE | 5 | 6 | 6 | 6 | OPC, Microglia |
| HD3_XENIUM | 6 | 6 | 7 | 7 | OPC |
| HD4_MERSCOPE (not scored by H4) | 0 | 2 | 2 | 4 | all |
| HD4_XENIUM (not scored by H4) | 5 | 5 | 6 | 7 | OPC, Microglia |

Classes passing of 7 (H4 needs ≥ 6 on HD1, HD2 and HD3, per platform; `$A/shadow/heldout/heldout_h4.csv`). Fold is ≥ 3 in every class and sample under every option; the AUROC decides.

- **Why OPC fails under (a).** The argmax OPC class holds every COP argmax call, which the v1 COP rule never emits as broad OPC (§5.2). OPC's AUROC is 0.54–0.66 under (a) on every sample; it is 0.60–0.89 under (b), 0.68–0.88 under (b′) and 0.70–0.95 under (c). Under (a) OPC fails by construction, so "≥ 6 of 7" becomes "all six other classes must pass".
- **"The assigned class" is an open choice for the user**, to be settled in the M3 PR before the definition is frozen; plan §5.8 does not define it, and M3 did not pre-register a reading. The options: (a) the argmax (strictest; HD2 both platforms and HD3_MERSCOPE fail); (b) the argmax with the §5.2 COP rule applied, COP calls that fail it staying at lineage and leaving the OPC class (HD2 both platforms fail); (b′) the same with the rule's SEA-AD confident-OPC rescue, whose SEA-AD calls saw the held-out genes (HD2_MERSCOPE fails); (c) WHB-only confident calls (HD2_MERSCOPE fails). Every option but (a) scores fewer cells than all table cells, so whichever the user picks is recorded as a definition in §2 before gate H, not as a loosening decided after the fact.
- Set c scores one class fewer (AQP4 is a set-c exclusion) and passes one class fewer on six of the eight samples under (a).

### 11.4 H16 prototype: realised flag rates

Confident cells, proseg_hybrid (`$A/shadow/flags/flag_rates.csv`). **Contamination:** 0.0–3.9% in every human class × platform stratum, 0.7–1.8% on MS1. Every stratum is informative except HD4_MERSCOPE fibroblasts, which have too few cells for a null. **Diffuse:** 0–44%. Ten of the 56 human strata exceed 15% and five exceed 30%: HD1_XENIUM oligodendrocytes 35% and microglia 44%; HD3_XENIUM oligodendrocytes 33%, microglia 35% and fibroblasts 30%. On MS1 the oligodendrocyte lineage is at 42%. These strata are marked uninformative. reseg is at ≤ 3.5% (contamination) and ≤ 4.4% (diffuse). Production flags are M4. H16's 15% and §4.3's 30% for the diffuse flag should be reconciled there.

### 11.5 LL value test (OD-B8) and the OD-B13 trigger

**The typer is the LL (vii) recipe on WHB-frontal bundle profiles**, not E1 (vii) itself: E1 `14_ll_contam.py` (`LLmix_wbctx+platform`) ported onto the WHB set-a bundle's 122 cluster profiles (17 superclusters, sinks included), where E1 used a 204-cluster whole-WHB cortex reference (plan §3.2 `ll_whb_ctx_profiles`). It typed every table cell of the 16 samples, once with factors capped at ±2 log2 (M10) and once uncapped (E1). Its agreement with E1 (vii) on E1's four 30k pilot subsets (`$A/shadow/ll/ll_vs_e1_vii.csv`):

| Pilot subset | Broad agreement, uncapped / capped | Supercluster agreement, uncapped / capped |
|---|---|---|
| HD1_MERSCOPE | 0.834 / 0.828 | 0.822 / 0.797 |
| HD1_XENIUM | 0.773 / 0.752 | 0.759 / 0.734 |
| HD2_MERSCOPE | 0.746 / 0.727 | 0.721 / 0.688 |
| HD2_XENIUM | 0.753 / 0.717 | 0.732 / 0.692 |

On the same cells the port's broad call agrees with WHB's argmax about as often as E1 (vii)'s does (0.71–0.82 uncapped, 0.65–0.82 capped, vs 0.68–0.81 for E1 (vii)), so the OD-B13 direction below does not depend on the port. Scoring OD-B8 on (vii) proper would need a rerun on `ll_whb_ctx_profiles`, which the store does not hold yet (§7, open question).

**OD-B8 has no pre-registered operationalisation.** The plan asks whether LL moves referee outcomes by > 2 points or coverage by > 5 points; how LL would be used is not stated. Two readings are reported; neither was fixed in advance (the first was chosen in `scripts/acceptance/shadow_ll.py` at M3 C2). OD-B8 goes to the user in the M3 PR with both.

| Sample (proseg_hybrid, capped) | (1) Coverage v1 → v1.1 (SEA-AD or LL below 60) | Bound: v1 without the below-60 rule − v1 | H10 v1 → v1.1 (ceiling gain) | Marker plausibility v1 → v1.1 | (2) LL as tie-breaker: referee gain | E2 LL rule − v1 | LL as the below-60 vote − v1 |
|---|---|---|---|---|---|---|---|
| HD1_MERSCOPE | 0.704 → 0.707 | 0.005 | 0.989 → 0.987 (+0.000) | 0.936 → 0.935 | +0.096 | −0.030 | −0.020 |
| HD1_XENIUM | 0.597 → 0.598 | 0.002 | 0.980 → 0.979 (+0.000) | 0.907 → 0.907 | +0.125 | −0.047 | −0.039 |
| HD2_MERSCOPE | 0.438 → 0.443 | 0.009 | 0.961 → 0.959 (+0.001) | 0.940 → 0.937 | +0.061 | −0.054 | −0.043 |
| HD2_XENIUM | 0.510 → 0.512 | 0.003 | 0.978 → 0.977 (+0.000) | 0.925 → 0.925 | +0.077 | −0.086 | −0.079 |
| HD3_MERSCOPE | 0.756 → 0.759 | 0.005 | 0.991 → 0.990 (+0.000) | 0.954 → 0.954 | +0.141 | −0.037 | −0.023 |
| HD3_XENIUM | 0.648 → 0.650 | 0.002 | 0.897 → 0.895 (+0.002) | 0.936 → 0.935 | +0.093 | −0.063 | −0.052 |
| HD4_MERSCOPE | 0.459 → 0.466 | 0.009 | 0.940 → 0.937 (+0.002) | 0.940 → 0.937 | +0.112 | −0.056 | −0.035 |
| HD4_XENIUM | 0.556 → 0.559 | 0.004 | 0.926 → 0.924 (+0.002) | 0.924 → 0.924 | +0.097 | −0.047 | −0.035 |

- **Reading (1), LL as the v1.1 below-60 vote: not met, and uninformative.** v1.1's confident cells (SEA-AD or LL agreeing below 60 counts) are a subset of "v1 without the below-60 rule" (the same ≥ 60 veto and rescues; checked on every sample, `v11_within_bound`). So the coverage gain is bounded by the below-60 SEA-AD cost, 0.2–0.9 points here and at most 1.0 on any dataset, far below 5; measured +0.1 to +0.6. H10 is at 0.90–0.99 with 10,000–51,000 disputes and v1.1 adds 204–561 cells, so even if every added cell were a won dispute H10 would rise by at most 0.2 points (`h10_ceiling_gain`), below 2. The test cannot pass by construction; its outcome says nothing about LL. The uncapped factors give the same.
- **Reading (2), LL as tie-breaker of WHB-vs-SEA-AD disputes: exceeds 2 points, but circular.** With LL deciding the disputes, markers side with the chosen label 6.1–12.5 points more often than with WHB alone on the development datasets (6.1–14.1 on all, capped; 7.7–14.5 uncapped). LL and the referee read the same marker counts, so this favours LL by construction, and it is not a v1.1 behaviour (WHB names every class, OD-B2).
- **For the user (M3 PR):** keep LL out of v1.1 (reading 1; `likelihood.py` stays for M10), or adopt it as a tie-breaker (reading 2) with a non-circular check first (held-out genes, §5.8), or score OD-B8 on (vii) proper once `ll_whb_ctx_profiles` is built.
- **OD-B13: not triggered.** The below-60 SEA-AD rule removes 0.2–0.9 points. The LL rules keep 0.6–8.6 points *less* coverage than the SEA-AD rule on every dataset (reseg included, both factor variants). The LL part of M10 stays after M8.
- For information: in WHB-vs-LL disputes (proseg_hybrid, capped) the referee is split, with WHB winning 0.36–0.60 of them and LL 0.22–0.45.

### 11.6 WHB vs SEA-AD glial JSD (information, OD-B2 decided)

Glial (astrocytes, oligodendrocytes, OPC, microglia) soft JSD, MERSCOPE vs Xenium, WHB − SEA-AD with a paired block-bootstrap CI (matched tiles, one draw for both sections and both references; §2), SEA-AD's soft composition on E2's class × subclass mass (§2): HD1 −0.025 [−0.027, −0.023], HD2 −0.038 [−0.047, −0.029], HD3 −0.021 [−0.023, −0.019], HD4 −0.061 [−0.064, −0.057]. WHB's glial composition agrees better across platforms on every pair. At 7 classes (soft) WHB is also better on every pair (−0.013, −0.032, −0.005, −0.016); before the review SEA-AD's soft mass lacked the class-level factor, which made its composition harder than WHB's root-level one and gave HD4 +0.011. The argmax 7-class difference on HD4 stays +0.031 [0.026, 0.037]. No decision is asked for.

### 11.7 Files

| Directory under `$A/shadow/` | Content |
|---|---|
| `e8/` | `runs/` (HD2 original_seg and proseg_mask, MS1 original_seg, proseg_mask and reseg MAP outputs), `metrics/e8_{human,human_jsd,mouse}.csv`, `REPORT.txt` |
| `x1/` | `x1_{factors,jsd,jsd_differences,referee,samples,compositions}.csv` (joint CIs and the independent sensitivity), `runs/` (rescaled WHB re-maps), `REPORT.txt` |
| `heldout/` | `heldout_{markers,enrichment,h4}.csv` (label sets `heldout_argmax`, `heldout_argmax_cop_rule`, `heldout_argmax_cop_rule_seaad`, `heldout_whb_confident`, `production_argmax`), `runs/` (held-out WHB re-maps for set a, set c and X1), `REPORT.txt` |
| `flags/` | `flag_{rates,nulls,scores,depth}.csv`, `REPORT.txt` |
| `ll/` | `ll_{sample_metrics,referee,factors,decision}.csv`, `ll_vs_e1_vii.csv` (the port vs E1 (vii) on the pilot subsets), `calls/` (LL calls per sample and factor variant), `REPORT.txt` |
| `glial_jsd/` | `glial_jsd.csv`, `glial_compositions.csv`, `REPORT.txt` |
| `superseded_20260927_pre_review/` | The pre-review CSVs and reports of every item (independent tile resampling, E1's SEA-AD definition, no float32 tolerance) and the pre-review `x1/runs`, `heldout/runs` |

## 12. Decisions of 2026-09-27 (M3b H18 follow-up)

The M3b self-map left H18 exceptions that were a matter of rule, not of the panels (`$A/m3b/resolvability/STAGE_C_REPORT.txt`, `$A/m3b/review_fix/REVIEW_FIX_REPORT.txt`): deep depth bins with ≥ 50 test cells but < 50 confident calls (set a Oligo at 120 counts, 42 calls, all correct), inherited by the bins beyond `D_max`, and human Vascular / Fibroblast / COP with only 23 / 5 / 20 held-out-donor test cells. The user decided (plan OD-E10, OD-E11):

1. **Pooled deep bins, one rule with gate P.** A (level, class) depth bin with fewer than 50 confident calls on its check set takes the verdict of the deep-end pool: from the deep end, bins are pooled into a "≥ d" set, each test cell counted once at its deepest bin, until the set holds 50 confident calls; its shallowest bin is D_P. The pool is judged as gate P judges a tested set (§9, "Evaluation rules"): point precision ≥ target and Wilson 95% lower bound (Kish effective n) ≥ target − 0.02, with coverage ≥ 0.2, the target of D_P, and `t*` fitted on the pool's fit half in the fitted regimes. The verdict is carried to every pooled bin, marked `resolvability_extrapolated`. A class that never reaches 50 confident calls even fully pooled stays `not_resolvable` (`insufficient_calls`). Implementation details (M3b, recorded here so gate P uses the same rule): a cell's deepest bin is taken over all its calls before any class or confidence filter; D_P keeps its own verdict when it holds 50 calls itself and is withdrawn when the pool fails (both tests cover it, as the gate-P tested sets do); the point estimate joins the §8.3 rule for single bins too; in RESOLVE a pooled set is reweighted as one set to the dataset's cells at ≥ D_P. Gate P keeps n_min = 200.
2. **Other-region non-neuronal test cells.** The human held-out test set tops every non-neuronal supercluster up to its cap (1,000) with WHB non-neuronal nuclei of E2's 14 neocortical dissections outside the frontal reference (MTG, STG, M1C, A43, A40, V1C, V2, A19, S1C, A1C, A5-A7, ITG, A38, A13; `research/insilico/01b_extract_nonneurons.py`), disjoint from every frontal reference, training and marker cell, recorded in the bundle (dissections, counts per class and donor).

**Re-measured** (resolvability version 3; `$A/m3b/followup/H18_FOLLOWUP.txt`): set a PREP H18 classes failing (broad / supercluster) 2 / 1 under version 2, 1 / 0 with decision 1 alone (broad OPC at 15), 2 / 1 with both decisions (broad Astro at 120 on power; broad and supercluster Oligo at 120, where other-region COP cells are called Oligodendrocyte); Vascular and Fibroblast are now emitted. MS2 passes; MS1 keeps the Immune subclass at 100 counts and, reweighted, the classes its datasets hardly contain. The H7 proxy (confident broad share after reweighting, an upper bound of H7) passes on every dataset with decision 1 alone and misses on HD1 MERSCOPE with both (.576 against ≥ .62; .638 with decision 1 alone). These are the H18 / H7 exceptions for the gate PRs; no threshold is changed.

## 13. Decisions of 2026-09-28 (M3b final follow-up and review 3)

M3b review 2 (resolvability version 4, `$A/m3b/review2/REVIEW2_H18.txt`) left two open points: 694 of the 2,955 other-region test cells were of clusters the held-out training reference lacks, and the rebuilt bundles redrew the simulation of every test cell because one random stream ran over all of them (0–7 of 85 validated broad / supercluster bins per human dataset moved; HD4 MERSCOPE's H7 proxy went from .311 to .297, below its .30 target). Two changes follow, neither of which changes a threshold, a target or a decision rule:

1. **Other-region cells only of training clusters** (user decision 2026-09-27, implemented 2026-09-28; plan OD-E11). The human held-out test set's other-region non-neuronal top-up draws only from clusters the held-out training reference keeps (≥ 5 training cells). The candidates left out are counted in `bundle.json` (`test_set.other_region`: `cluster_rule`, `n_excluded_cluster_not_in_training` per supercluster, and a check that no drawn cell is of an absent cluster). `HO_OTHER_REGION_VERSION` 2 enters the held-out and self-map build hashes.
2. **Keyed simulation draws** (robustness fix, resolvability version 5, corrected in version 6; plan §8.3 step 3). Each simulated cell's thinning and spill thinning come from a generator seeded by a stable hash of the seed, the recipe name and version, the cell id and the depth, and the spill partner is chosen by rendezvous hashing (uniform over the eligible cells). Adding, removing or reordering test cells then leaves every other simulated cell unchanged, except that a host whose partner leaves (or is outranked by an added cell) gets a new spill. The per-gene efficiency stays the pre-registered draw (one generator seeded by `[0, 0x6566]` over the panel's gene order), which never depended on the test cells. *Correction (M3b review 3, 2026-09-28):* the version-5 build had also keyed each gene's efficiency by the gene id. Neither change needed that, and it replaced the pre-registered seed-0 realisation with an unrelated one (correlation of the log efficiencies −0.06 on set a); resolvability version 6 restores the pre-registered draw and keeps the per-cell keys. With it the recipe (LogNormal(0, 0.8) efficiency in its seed-0 realisation, 25% spill, seed 0) is unchanged.

**Re-measured** (resolvability version 6; `$A/m3b/final_followup/FINAL_FOLLOWUP.txt` §12, `$A/m3b/final_followup/review3/H18_V6.txt`, `EXCEPTIONS_V6.txt`). The set a test set holds 11,040 cells (8,777 donor + 2,263 other-region; Vascular 412, Fibroblast 144, COP 152). Set a PREP fails H18 at broad Oligo 120 counts (79 calls, precision .911, Wilson bound .828 < .88); version 4 failed broad Astro, broad and supercluster Oligo, and supercluster Astro at 120. Reweighted to the eight human datasets, 23 (level, class) pairs fail against 25 under version 4: broad OPC on all eight (at 15 counts, and at 120 on five), broad Oligo on seven and supercluster Oligo on six (other-region COP cells of training clusters called Oligodendrocyte, 9–26% of the called set's confident weight), and broad and supercluster Immune at 60 on HD4 MERSCOPE. MS2 PREP passes; MS1 PREP fails the Immune subclass at 100 in this draw (as in version 4; it passes with seeds 1 and 2); reweighted, MS1 fails 0–2 classes or subclasses at 250 counts per segmentation (05 OB-IMN GABA, 09 CNU-LGE GABA; few positive-weight calls or small Kish n). The H7 proxy (confident broad share after reweighting, an upper bound of H7) is HD1 M .683, X .605; HD3 M .739, X .656; HD2 M .354, X .523; HD4 M .298, X .557: it misses on HD4 MERSCOPE (≥ .30 asked), as under version 4 (.297). These are the H18 / H7 exceptions for the gate PRs; no threshold is changed.

**Version 5's numbers are not the outcome of these changes.** With its redrawn efficiency, 15 reweighted pairs failed, set a PREP passed and the H7 proxy passed on all eight datasets (HD4 MERSCOPE .380). Those results come from the unrequested redraw; they are one more draw of the recipe, not the result of the two changes, and the gate PRs should not cite them as passes.

**The verdicts depend on the simulation draw** (`$A/m3b/final_followup/review3/FACTORIAL_ANALYSIS.txt`). A 3 × 3 grid on set a crossed the per-cell seed (thinning, spill, partner) with the efficiency seed, one realisation per grid cell, mapped at seed 0; the additive model's residual holds the interaction and the mapping noise. Across the grid, HD4 MERSCOPE's H7 proxy spans .297–.431 and is below .30 in 2 of 9 cells, both with the seed-0 efficiency; HD2 MERSCOPE spans .320–.444 (below .34 in 1); the other datasets stay at least .028 above target. The reweighted H18 failures number 12–26 and set a PREP's 2–8. Changing only the efficiency seed moves on average 6.3 of the 83 validated broad / supercluster bins per dataset (1–14); changing only the per-cell seed moves 4.4 (0–9), both 6.5. Mapping the same simulated cells in another order (MapMyCells draws its bootstrap per chunk of query cells) moves 1.4 (0–2), yet that alone lifts HD2 MERSCOPE's proxy from .354 to .433 and adds 4 set a PREP failures. For the H7 proxy and the H18 counts neither factor is significant in this grid (F(2, 4) below the .05 critical value 6.94, except HD2 Xenium's H7 proxy, whose range is .007), and the residual is the largest component for most metrics. Only broad OPC at 15 clearly follows the efficiency draw: it fails on all eight datasets with the seed-0 efficiency under every per-cell seed, and on 0–5 with seeds 1 and 2 (F = 18.1, p = .01). The version-5 statement that the single efficiency draw moves these verdicts more than the per-cell draws do therefore holds for broad OPC at 15 and, on average, for the number of bins that move; it does not hold for the H7 proxy or the H18 counts. How the gate treats this spread (the pre-registered seed-0 draw, several draws, or the worst of them) is a pre-registration choice left to the user; until then no H18 / H7 verdict is recorded as a pass.

## 14. Decisions of 2026-09-27 (after the M3 and M3b PRs; recorded at M4)

These decisions change no threshold, target or decision rule in §9. Plan: OD-B7, OD-B8 and OD-B14 in "Open decisions for the user", D-A5, §7.4 and the §14 human flip rule.

1. **Default segmentation (OD-B7): proseg_hybrid for human, reseg for mouse.** This is the M3 E8 recommendation (§11.1): human keeps proseg_hybrid (reseg has purer cells but 17–43% fewer confident cells per mm²); mouse MS1 uses reseg (most confident subclass cells, lowest foreign fraction). The registered criteria keep the segmentations they name (§9: human on proseg_hybrid with reseg held out for H17; MO1 on MS1 proseg_hybrid and MS2 original_seg); the decision sets the pipeline default, not a criterion. MS2 is re-checked after the M0a re-run.
2. **No likelihood-typer vote in v1.1 (OD-B8).** LL does not join the v1.1 consensus in either reading of §11.5. `merxen.annotation.likelihood` and the LL (vii) recipe stay for diagnostics. The §5.3 degraded-mode truth table keeps its LL rows (`consensus.DEGRADED_MODES`), but no production configuration enables them. OD-B13 stays not triggered (§7 item 2).
3. **H2, H4 and the H7 / H18 exceptions are decided at M8 (OD-B14).** The baseline H2 failures (HD4_MERSCOPE proseg_hybrid 1.02%, HD2_MERSCOPE reseg 1.04%; §7 item 1), H4's open "assigned class" and its HD2 failures (§7 item 7, §11.3), and the H7 / H18 exceptions of §12–§13 (HD4 MERSCOPE's H7 proxy .298; the reweighted H18 failures) are re-measured at M8 with M4's final RESOLVE rules: statuses, resolvability-gated emission reweighted to each dataset, and the COP rule. The M8 gate PR brings, for each, a fix or a written exception proposal for the user's approval. Until then no H2, H4, H7 or H18 verdict is recorded as a pass, and no threshold is loosened (§1).

## 15. Re-measured with M4's final rules (2026-09-28; recorded at M4, no decision)

OD-B14 (§14 item 3) asks for H2, H4 and the H7 / H18 exceptions to be re-measured with M4's final RESOLVE rules before M8. Stage D of M4 ran `merxen annotate-resolve` (commit `9458f89`: RESOLVE statuses, resolvability-gated emission reweighted to each dataset with the store's version-6 bundles, floors, gate, COP rule, flags) on the M3 shadow MAP outputs of all eight human datasets, proseg_hybrid and reseg, and re-scored the criteria with `scripts/acceptance/resolve_criteria.py` (`c31cc6f`). **No threshold, target or decision rule is changed here.** Full tables, causes and the candidate fix or exception text for every failure are in `$A/m4/CRITERIA_AFTER_M4.txt`; the per-criterion rows in `$A/m4/criteria/criteria_table.csv`; the M4 exit checks in `$A/m4/STAGE_D_REPORT.txt` and `$A/m4/shadow/`. *M4 review (2026-09-28):* the COP flag, the consensus tier codes, the SEA-AD subclass status order and margin, the `per_platform` pair statistics and the exit accounting were corrected, and stage D was re-run at `2403e17` (criteria at `d24c1f6`); the lineage, broad, NT and supercluster statuses and names, the final labels, the pair JSDs and every criterion value below are unchanged (`$A/m4/review_fix/REVIEW_FIX_REPORT.txt`, `label_changes.csv`).

| Criterion (flip-rule datasets) | M4 result | Pass |
|---|---|---|
| H1 soft JSD | HD1 .129 [.116, .143], HD2 .133 [.126, .142], HD3 .126 [.112, .143], HD4 .227 [.212, .242] (identical to §3–§4) | yes |
| H2 implausible ≤ 1% | HD4_MERSCOPE 1.02% (unchanged: `flag_implausible` follows the assigned node, not the statuses); every other dataset 0.30–0.91% | **no** (HD4_M) |
| H3 WHB–SEA ≥ 20 counts | .926 / .917 / .825 / .908 / .948 / .929 / .828 / .921 (unchanged) | yes |
| H4 held-out genes | Headline, non-circular (plan §5.8: no method saw the held-out genes): M4's rules on the held-out WHB re-map in the WHB-only mode (`m4_resolve_heldout_whb_only`): HD1 7 / 7, HD3 6 / 7, HD2_XENIUM 7, **HD2_MERSCOPE 4** (Oligodendrocytes, Microglia, Vascular fail on AUROC at the ceiling their held-out detection allows; fold 6–14). Secondary, partly circular (the production SEA-AD votes saw the held-out genes; below 60 counts SEA-AD agreement decides confidence): `m4_resolve_heldout`, the same counts. The "assigned class" is still the user's choice (§11.3); the M3 options (a)–(c) give the §11.3 values | **no** (HD2_M) |
| H5 COP control | confident COP supercluster 0 (COP is never emitted), confident broad OPC 0.3–2.1% | yes |
| H7 broad coverage | HD1 .677 / .591, HD2 .345 / .504, HD3 .734 / .641, **HD4_MERSCOPE .288** / .509 | **no** (HD4_M) |
| H8 gate | levels as expected, none `failed`; HD4_M warns as expected; **HD2_MERSCOPE also warns** (.120 of segmented objects; M3 .152, §7 item 3) | levels yes; warning deviates |
| H9 marker plausibility | HD1 .997 / .996, HD2 .991 / .987 | yes |
| H10 referee | .990 / .981 / .960 / .980 | yes |
| H16 flag transparency | every flag × class × platform stratum reported; all strata above 15% marked | yes |
| H17 reseg (HD1, HD2) | H1 + .02: .127 / .125; H9 .999 / .999 / .991 / .985; **H2 HD2_MERSCOPE 1.04%** | **no** (H2) |
| H18 | not re-measured (M3b §13: set a PREP broad Oligo 120; 23 reweighted pairs); RESOLVE's label changes in `$A/m4/shadow/resolvability_exceptions.csv` | **no** (as §13) |

- **Common cause of the H7 failure, the extra H8 warning and most of H18.** Without resolvability tables RESOLVE reproduces the M3 shadow coverage exactly. With them, broad Oligodendrocytes are not emitted at 10–60 counts on HD4_MERSCOPE (0 confident broad Oligodendrocytes, against 14.8% of table cells without the tables) and at 10 counts on HD2_MERSCOPE (15.8% vs 24.0%), and broad OPC at 10–15 counts on all eight datasets. In the set a self-map almost every wrong broad Oligodendrocyte call is a COP test cell called Oligodendrocyte (COP's broad class is OPC; the other wrong calls are at most 0.3% of any bin's weight); reweighting to MERSCOPE datasets with a large low-depth COP share turns that into `wilson_bound_below_target`.
- **Candidate fixes for the M8 gate PR (none adopted).** First, M3b's option, which keeps the truth definition: drop or down-weight the other-region COP test cells, which cause most of the wrong broad Oligodendrocyte calls (not measured at M4). Second, measured (`$A/m4/shadow/cop_counterfactual_*.csv`, `cop_counterfactual_prep.txt`): scoring a COP test cell called Oligodendrocytes as correct at the broad level removes every broad Oligodendrocyte exception (set a PREP broad Oligo 120: precision .911 → 1.000, emitted) and gives broad coverage HD2_M .427 and HD4_M .436 (H7 passes), within ±0.05 of M3 on every proseg_hybrid dataset; broad OPC at 10–15 stays not resolvable, so HD2_MERSCOPE keeps the warning (.148). **This second option changes the truth definition** (COP's broad class is OPC in the vocab) and makes the §8.3 / H18 emission test easier, so it is a loosening under rules 1–2 and needs the user's written approval. RESOLVE's COP rule does not justify it: the rule applies only to WHB COP calls, so a real COP cell that WHB calls Oligodendrocyte gets a confident broad "Oligodendrocytes" label from RESOLVE, wrong under the vocab, and the counterfactual stops penalising exactly that error (M4 stage D first gave that justification; the M4 review of 2026-09-28 withdrew it). The remaining option is written exceptions.
- **Definitions added or differing (rule 4).** H9, not defined at M3 (§8), is scored with ports of the two dataset reports' pseudo-label methods (`scripts/acceptance/marker_pseudo_labels.py`; they reproduce the reports' stored labels exactly and the legacy agreement 62.6% / 45.2% of `data/HD1` §C): the share of confident broad cells with a resolved pseudo-label whose six-class label matches, with the HD1 §C method on HD1 and the HD2 §4 method on HD2. H8's warning is reported as its own row. H16 marks every stratum above 15% (`informative_h16`); the flags themselves follow §4.3 (null above 15% for contamination, above 30% for the diffuse flag), so five diffuse strata between 15% and 30% keep their flags. H4's headline is the non-circular WHB-only label set; the SEA-AD-voted set is a secondary, partly circular row (M4 review). `flag_cop_suppressed` follows plan §4.3 (a COP call on a confident lineage whose COP rule fails, whichever broad check decides the status), the M3 shadow count (§3–§5: 0.06–3.5% of table cells); stage D had first set it only when the COP rule was the first failing check (0.00–0.08%); it is reported, never scored. `ct_consensus_tier` 0 means a confident disagreement only (plan §4.1); no informative method is −1 (stage D had written 0 for both). The SEA-AD subclass follows the documented status order (its SEA-AD agreement check last) and its margin is on the raw (aggregate probability) scale. None of these changes a scored value.
- **Rule deviation recorded (rule 4; M4 review, 2026-09-28).** Plan §5.2 rule 4 asks a confident broad for the supercluster; M4 also asks a confident NT for neurons, because `ct_final_level` is the deepest level of a contiguous confident chain (lineage → broad → NT → supercluster; the schema's final-chain check). It withholds 12–74 neuron supercluster labels per full-gate proseg_hybrid dataset (HD1 M / X 47 / 61, HD2_X 23, HD3 M / X 12 / 35, HD4_X 74; reseg 3–15; review rule check, resolvability not applied) that pass every other rule 4 check; it can only lower supercluster coverage.
- **Exit accounting corrected (M4 review).** A label lost because its parent became unresolved now takes the category of the chain level where the cell was first lost (same depth bin): stage D had counted all of them as allowed by the exit, which allows only classes with fewer than 50 test cells and extrapolated bins. The follow-on losses of the broad Oligodendrocyte / OPC and lineage Fibroblast / Oligodendrocyte exceptions are exceptions too: on proseg_hybrid broad 96 and supercluster 10,582 (Oligodendrocyte 10,154, COP 341, OPC 87) of the 10,734 labels stage D listed as allowed; 56 follow a parent with fewer than 50 test cells (reseg: 2,246 exceptions, 34 allowed; `$A/m4/shadow/resolvability_exceptions.csv`, `via_parent`). The exit verdict (not met) does not change.

## 16. Mouse `avg_correlation` floor: shadow study (M6; pre-registered 2026-09-29, before it runs)

Plan §7.3: "`avg_correlation` is recorded (`ct_class_corr`); the M6 shadow evaluates floors 0.4 and 0.5 against coverage and marker consistency by depth and pre-registers one." This section fixes the design, the metrics and the selection rule **before** the study is run. It changes no §9 threshold, target or decision rule: MO1 scores the MapMyCells class of every marker-pseudo-confident cell, which a confidence floor does not change, and a floor can only remove confident class calls.

- **Datasets and inputs.** MS1 proseg_hybrid (103,736 table cells) and MS2 original_seg (112,158), the MO1 datasets. MAP: the M6 stage A outputs with the region step (`$A/m6/stageA/map/{ms1,ms2}`; bundles `5a032858` / `daa8c4a6`; pruned calls). RESOLVE: `merxen annotate-resolve` with the M6 mouse rules (§7.3: class bootstrap probability ≥ 0.90, class floor 20 counts, not region-dropped without a confident re-map, resolvability-gated emission) and **no** correlation floor. `ct_class_corr` is the class-level MMC `avg_correlation` of the pruned primary run.
- **Candidates.** f ∈ {none, 0.40, 0.50}. Under floor f a cell is class-confident when `ct_class_status == confident` and `ct_class_corr ≥ f` (float32 tolerance of `schema.meets_threshold`).
- **Marker pseudo-labels (fixed, MO1's own).** MS1: `data/mouse_ms1/out/tables/C_pseudo_labels_per_cell.csv` (`pseudo_fine`), scored with `expected_pseudo_for_mmc` of `data/mouse_ms1/scripts/ms1_sections_extra.py` (ported verbatim; `unassigned` cells are not pseudo-confident). MS2: `data/mouse_ms2/tables/C_pseudo_labels_per_cell_original_seg.csv.gz` (`pseudo_class` in Glut, GABA, Astro, EpendCP, Oligo, OPC, Immune, Endo, Mural, VLMC), scored with the six-group map of `review_scientific/mouse_marker_consistency.py` (ported verbatim). Check before any floor is scored: MO1 over all marker-pseudo-confident cells with the **unpruned** MAP labels must reproduce .9114 (MS1) and .8556 (MS2) to three decimals; the pruned-label MO1 is reported beside it. If the check fails, the study stops and is reported without a selection.
- **Metrics** per dataset and floor, overall and per depth bin of total counts ([20, 50), [50, 100), [100, 250), [250, 500), [500, 1000), [1000, 2000), ≥ 2000):
  - `cov(f)`: share of table cells class-confident under f;
  - `cons(f)`: marker consistency among marker-pseudo-confident cells class-confident under f (MO1's confident-cell value);
  - `rem_cons(f)`: marker consistency among marker-pseudo-confident cells confident under `none` that f removes, with their number `n_rem(f)`.
- **Admissible floor.** f ∈ {0.40, 0.50} is admissible when, on **both** datasets: (i) `cons(f) − cons(none) ≥ 0.005`; (ii) `n_rem(f) ≥ 50` and `rem_cons(f) ≤ cons(f) − 0.10` (the removed calls are markedly less marker-consistent than the kept ones); (iii) `cov(none) − cov(f) ≤ 0.02` overall and `≤ 0.05` in every depth bin holding ≥ 1,000 table cells.
- **Selection.** No admissible floor: none is adopted (`ct_class_corr` stays report-only; the v2 calibration of §7.3 revisits it). One admissible floor: that floor. Both: 0.50 when, on both datasets, `cons(0.50) − cons(0.40) ≥ 0.005` and `cov(0.40) − cov(0.50) ≤ 0.01`; otherwise 0.40.
- **Application of the selected floor.** It becomes `AnnotationThresholds.wmb_class_min_corr`, a `low_confidence` check at the mouse class level (and so below it) for the real-data-validated mouse families only (MS1, MS2), where it was measured. `avg_correlation` depends on the number of markers (§7.3), so other panels take no floor until the resolvability simulation provides one (M10); RESOLVE records the floor as not applied for them.
- **Outputs.** `scripts/acceptance/mouse_corr_shadow.py`; tables and a plain-text report in `$A/m6/stageB/corr_shadow/` (the run log records the git commit and the script's sha256). The outcome is recorded below this section, with the numbers, once the study has run.

**Outcome (run 2026-09-29, after this section was committed in `2cefe69`; script `scripts/acceptance/mouse_corr_shadow.py` at `b55823f`, sha256 `eb0bcf03…`; `$A/m6/stageB/corr_shadow/CORR_SHADOW_REPORT.txt`, `corr_floor_{ms1,ms2}.csv`, `corr_shadow.json`).** The label tables are the M6 RESOLVE outputs at `b55823f` (`$A/m6/stageB/resolve_final/{ms1,ms2}`). The MO1 check passes: .9114 (MS1, 88,931 cells) and .8556 (MS2, 82,857) with the unpruned labels; .9118 and .8558 with the pruned ones. **Selection: none** — neither floor is admissible, so `wmb_class_min_corr` stays `None` and `ct_class_corr` is report-only.

| Dataset | Floor | Class coverage | `cons` (confident calls) | Removed (pseudo-confident) | `rem_cons` | Failing clauses |
|---|---|---|---|---|---|---|
| MS1 | none | .7321 | .9630 | – | – | – |
| MS1 | 0.40 | .7297 | .9636 | 167 | .701 | (i) gain +.0006 |
| MS1 | 0.50 | .6552 | .9756 | 5,785 | .826 | (iii) coverage −.077; bins [20, 500) lose .08–.22 |
| MS2 | none | .8249 | .9030 | – | – | – |
| MS2 | 0.40 | .8181 | .9033 | 444 | .847 | (i) gain +.0004; (ii) removed .847 > .803; (iii) bin [100, 250) −.052 |
| MS2 | 0.50 | .7233 | .9103 | 8,485 | .852 | (ii) removed .852 > .810; (iii) coverage −.102; bins [100, 1000) lose .13–.31 |

A floor of 0.40 removes few calls (0.2–0.7% of confident cells) and changes consistency by less than a thousandth; 0.50 buys 0.7–1.3 points of consistency at 7.7–10.2 points of class coverage, mostly below 500 counts, and on MS2 the calls it removes are nearly as marker-consistent as those it keeps (.852 vs .910). Consistency falls with depth on MS2 (.854 at 1,000–2,000 counts, the deep doublet-like segments), which no correlation floor addresses. No threshold, target or decision rule of §9 changes.

## 17. H12 in the report: the depth-input check and the CI unit (M7 review; recorded 2026-09-30, before M8 scores H12)

Plan §9 item 9 and §14 H12 fix the criterion ("medians with non-overlapping 95% CIs on both platforms"; "oligodendrocytes WM > GM on both platforms") but not the depth input's validity or the bootstrap unit. The M7 review found both open questions mattered on the development pairs. This section records how the report (`merxen annotation-report`, `report_depth`) computes H12 from M7 on. **No threshold, target or decision rule of §9 changes.**

- **Depth-input check (label-free, per pair).** In the 200 µm bins of the Xenium frame holding ≥ 5 table cells of each section, the two sections must agree on cortical-ribbon membership (more than half of a bin's cells inside the ribbon) in ≥ 80% of the bins, and their per-bin mean equivolumetric depths must correlate with Pearson r ≥ 0.8. When they do not, the platform whose depth was computed on transformed coordinates (its depth QC `coordinate_source` names an `_aligned` shape; both platforms when neither or both did) is invalid: every H12 record of that platform, and the pair's replication, is `not_available` with the reason `depth_input_invalid`. A single platform, or coordinates not in one frame, is not checked. The check can only withhold H12; it never makes it pass.
- **Why, and the calibration.** The published MERSCOPE proseg_hybrid depth of HD1 and HD2 (legacy `compute_cortical_depth` and `compute_cortical_depth_mapfirst`) was computed on the aligned (Xenium-frame) cell coordinates, while the manual pia / WM boundaries are drawn in the MERSCOPE native frame (`cortical_depth.pipeline._resolve_shape_key` prefers `<key>_aligned_nonrigid` on MERSCOPE): its ribbon cuts across the tissue. Check values: HD1 ribbon agreement .530 and depth r .144, HD2 .473 and −.114. The legacy reseg depth of the **same** cell ids, computed on `obsm:spatial` (native frame), gives .966 / .987 and .938 / .978; Xenium reseg vs Xenium proseg_hybrid (same section, one frame) .999 / 1.000 (`$A/m7/review_fix/depth_validity_calibration.txt`). The published MERSCOPE H12 values of both pairs (the HD1 ordering failure, oligodendrocyte WM − GM −.013 / −.036, the between-platform replication) therefore describe the broken input, not the labels.
- **CI unit.** The primary 95% CI of every H12 median resamples 500 µm tangential blocks, `floor(tangential_position_um / 500)`: each block is a full pia-to-WM strip, so every replicate keeps the depth slices in their anatomical proportion, and holds about ten neighbouring streamlines (about 50 µm apart), so the blocks are not near-duplicates. The square 500 µm tile bootstrap (§5.5's composition unit) is reported beside it as a conservative sensitivity (`kind = square_tile_500um`): a tile covers about a fifth of the cortical depth, so resampling tiles re-mixes depth slices. The per-streamline `column_id` bootstrap reported in the first M7 build (M7 stage A, D2) is dropped: its ~50 µm strips are strongly autocorrelated, so its CIs are too narrow. The WM − GM contrast keeps square tiles (white-matter cells have no tangential position). `acceptance_metrics.json` records the unit (`depth_ci_method`).
- **This choice was made after results were seen.** On HD1_XENIUM the ordering fails with tile CIs (Deep-layer IT [.720, .815] vs Deep-layer NP/CT/6b [.812, .865]) and passes with 500 µm tangential blocks ([.741, .791] vs [.824, .857]; 1000 µm blocks [.728, .805] vs [.814, .867]). The strip CI is narrower, so under rule 2 adopting it for the gate is treated as a loosening: **the M8 gate PR needs the user's written approval to score H12 on the tangential-block CI**; without it the square-tile result, reported beside it, is the one scored.
- **Consequence for M8.** H12 on HD1 and HD2 MERSCOPE is `not_available` until the MERSCOPE depth is recomputed in the native frame (a data-integrity fix to `main`, then the legacy and map_first depth runs re-run on a copy). As a diagnostic only (not a gate input), the report on a copy of the published depth table with the native-frame reseg depth of the same cells substituted gives: HD1_MERSCOPE ordering passes under both CI methods (Upper-layer IT .238 [.200, .300] < Deep-layer IT .656 [.617, .684] < Deep-layer NP/CT/6b .815 [.786, .851]), replicated with Xenium (Spearman .98 of the supercluster medians), oligodendrocyte WM − GM +.534 [.495, .567] (Xenium +.535); HD2_MERSCOPE (broad-only: no ordering) WM − GM +.514 [.469, .555] (Xenium +.620) (`$A/m7/review_fix/reports_native_depth_diagnostic/`).

## 18. Decisions of 2026-09-30 (M8 gate decisions, OD-B14; recorded before any M8 measurement)

OD-B14 (§14 item 3) sent H2, H4 and the H7 / H18 exceptions to M8, to be decided on M4's final rules (§15). The M8 decision memo (`$A/m8prep/M8_DECISION_MEMO.txt`; every number in `$A/m8prep/MEASURE_REPORT.txt`) put nine questions, D1–D9, with a recommendation for each; its dated CHECK section (same file) recomputed the numbers and corrected the memo (C1–C11), and its corrections supersede the memo sentences they quote. **The user's decisions, given in the chat on 2026-09-30:**

- on the M7 reports of HD1 and HD2 (the `$A/m7/REVIEW_GUIDE.txt` checklist): "I am happy on all those points. All looks correct to me in terms of what is being presented";
- on the memo: "With M8 decisions, I agree with all the recommendations and approve them".

Every decision below is therefore the memo's recommendation. **No threshold or target of §9 changes.** D1 changes a registered measurement input and is treated like a loosening (caveat below); D9 is a tightening; D2 and D8 keep the registered rule; D3, D4, D5 and D7 are written exceptions; D6 is the §11.3 definition choice.

| # | The memo's question (verbatim) | Decision | Label (§1 rules 1–4) |
|---|---|---|---|
| D1 | "Other-region COP test cells: drop / COP-as-correct / neither." | drop, implemented as the rebuilt self-map | changes no threshold or truth definition; changes the registered test set (§12 item 2) in the passing direction, so treated like a loosening (C1); approved in writing |
| D2 | "Simulation draw: seed-0 / majority / pooled / any-of-9 / v5." | seed-0, spread reported | no change (memo rule R0) |
| D3 | "[HD2]_M H8 warning exception: yes / no." | yes | exception |
| D4 | "H18 exceptions (broad OPC 15 + 120, Immune 60 on [HD4]_M, would-raise list): yes / no." | yes | exception |
| D5 | "H2: exception / vocab / metric." | exception | exception |
| D6 | "H4 assigned class: WHB-only / SEA-voted / M3 (a) / (b) / (c)." | WHB-only | definition (§11.3; not a loosening) |
| D7 | "H4 on [HD2]_M: exception / detection-aware / >= 30 counts." | exception | exception |
| D8 | "H16 diffuse null: 30% / 15%." | 30% | no change (plan §4.3) |
| D9 | "Supercluster NT-parent tightening: accept / literal." | accept | tightening (rule 2) |

**Caveat C1 (D1), stated with the approval.** The drop changes no threshold and no truth definition, which is how §15 frames M3b's option, but it edits a registered measurement input: the §12 item 2 held-out test set, which §9 records as part of the H18 measurement. The edit is made after the results are known, and it makes H18 and (through resolvability) H7 easier to pass. It also removes supercluster COP from H18's scope: supercluster COP falls below 50 test cells (only the held-out donor's 20 remain). No requirement is lost with it, because under version 6 floor 120 = D_max 120 left COP's expected set empty (CHECK S3), but rule 2 lists dropping a class, so it is said here. Under rule 2 the drop is therefore treated like a loosening, and the user's written approval quoted above is quoted again in the gate PR. The scientific rationale is weaker than the memo first stated (C2): 5–14% of the other-region COP test cells get confident broad Oligodendrocyte calls at 10–120 counts, and so do 1 of 20 donor COP cells at 30 counts and 1 of 17 at 60; per depth the two rates do not differ (Fisher p = .36–1.00). The other-region cells cause most of the wrong calls mainly because they are 87% of the COP test cells. With the drop, the COP → Oligodendrocyte error is estimated from 20 cells, so the check on that error (a real COP cell that WHB calls Oligodendrocyte gets a confident broad label) is weaker. The gate PR justifies the drop from the PREP self-map evidence only (PREP broad Oligo precision .911–.970 with the cells, .994–1.000 without), never from HD4's result (rule 3).

**D1 as implemented (M8 stage A1).** Every human self-map on the held-out test set (`whb_frontal_supc_clus_ho`: the WHB set a and set c bundles and the SEA-AD set a bundle) leaves out the test cells whose `test_source` is `other_region` and whose truth supercluster is Committed oligodendrocyte precursor (set a: 132 cells, 5,510 rows of the stored cells table; per depth 132 / 132 / 128 / 96 / 50 / 13). The held-out bundle and its `test_cells.h5ad` are unchanged; the cells are left out when the self-map loads them, as the M8-prep rebuild did (`$A/m8prep/scripts/m8_resim.py`, variant `resimdrop`): the remaining cells are simulated with the version-6 keyed draws, so only a host whose spill partner was a dropped cell gets a new spill (0.46% of R1 calls and 2.2% of bp values differ from the table-level drop), and mapped with the production configuration. The test-set revision (`HO_SELF_MAP_TEST_SET_REVISION` 1) and its rule enter every human self-map bundle's `build_hash` through the test-set params, so the rebuilt bundles land in new store build directories; `resolvability_summary.json` and `bundle.json` record the revision, the rule and the cells left out per supercluster and source. `RESOLVABILITY_VERSION` stays 6: version 7 is M3c's unmerged ensemble version, and a bump would also rehash the mouse bundles (CHECK S2). `HO_OTHER_REGION_VERSION` stays 2, because the held-out bundle's own top-up is unchanged. It reverses the other-region top-up decision of 2026-09-27 (plan OD-E11) for the COP supercluster only. The code applies the rule to every self-map whose test set is `whb_frontal_supc_clus_ho` (`reference.self_map_test_cells`), so it would also reach later human families on that test set (5K, the M13 custom panels); the approval covers the M8 bundles only (below, "Not covered"). Side effects the CHECK measured at table level (S1, S4): the SEA-AD set a self-map (bundle `d70cda25`) emits broad Oligo 120 and 250 (precision .902 / .910 → .981 / .990; would-raise stays 8); the set c WHB self-map (`5137090d`) changes no status (would-raise 6 → 4); reseg follow-on exception losses rise (2,246 under version 6, 2,672 with the rebuild). The H7 effect of the SEA-AD change is unmeasured until M8 runs.

**Measured with D1 at M8 prep (rebuilt self-map, seed-0 draw; to be re-measured by the M8 run).** H7 passes on all 8 datasets (HD4_MERSCOPE .288 → .436, HD2_MERSCOPE .427); set a PREP passes H18; 10 reweighted H18 (dataset, level, class) pairs fail (23 under version 6); broad coverage within ±0.05 of M3 on all 8 proseg_hybrid datasets (worst −.023); HD1_MERSCOPE confident broad OPC 2.06% → 1.01% (H5 ≤ 10% still passes); labels lost to resolvability 9,919 direct / 1,983 follow-on (version 6: 34,564 / 10,678).

**Approved exception texts** (the memo's recommended texts as corrected by the CHECK section; the gate PR updates each number with the M8 run's value and keeps the scope):

- **D3, H8 warning on HD2_MERSCOPE** (`MEASURE_REPORT.txt` §6 draft, Path A): "[HD2]_MERSCOPE carries the segmented-object coverage warning (.148 of 296,487 segmented objects with the other-region COP test cells dropped; .120 with the pre-registered test set; M3 shadow .152; threshold .15) in addition to its expected broad_only level. The remaining gap is broad OPC at 10–15 counts, which resolvability withholds on every human dataset. The warning is informational and never lowers the level; [HD2]_MERSCOPE is already restricted to broad-level statistics (OD-B6)."
- **D4, H18** (the §6 Path A draft with the rebuild's numbers, C11): "Broad OPC is not emitted at 15 counts on the eight human datasets (self-map precision after reweighting .79–.87, Wilson bound .74–.83 < .88: held-out-donor neurons and oligodendrocytes called OPC, with the COP test cells' correct OPC calls capped by the judged-set weight trim), nor at 120 counts on seven of them (precision .87–.92, Wilson bound .80–.87; [HD3]_XENIUM emits it at .935 / .892; COP keeps 1 test cell at 120 counts once the other-region COP cells are dropped). Broad and supercluster Immune are not emitted at 60 counts on [HD4]_MERSCOPE (precision .911 / .842, Wilson bound .859 / .774: held-out-donor deep-layer IT neurons called Immune); no [HD4]_MERSCOPE label is lost there (no confident Immune call in that bin even without the tables). The OPC cells keep their confident oligodendrocyte lineage except at 10 counts on the three datasets where lineage OPC is withheld too (2,053 cells)." The would-raise list is part of D4: every validated threshold the local rule would raise is listed in the gate PR (at M8 prep: set a PREP 15 with the drop, 19 without; SEA-AD set a 8; set c 6, 4 with the drop at table level). The gate PR also cites the remaining resolvability-exception rows and their follow-on losses (memo item 4; rebuild numbers, proseg_hybrid): broad OPC 10 / 15 / 120; lineage OPC 10 (3 datasets, 2,053 cells); lineage Fibroblast 10 (4 Xenium datasets, 72 cells); lineage Oligo 120 (HD4_XENIUM, 28); supercluster Exc 10 (6 datasets, 140); follow-on losses broad 880, supercluster 1,103.
- **D5, H2** (`$A/m4/CRITERIA_AFTER_M4.txt` §2.1): "H2 exceeds 1% on [HD4]_MERSCOPE proseg_hybrid (1.02%) and [HD2]_MERSCOPE reseg (1.04%), both shallow MERSCOPE sections (median 16–17 counts). The implausible calls are cortical neurons that WHB places in the Miscellaneous / Splatter sinks or the Amygdala excitatory supercluster (SEA-AD: L5 ET, Sst Chodl, L6 / EC IT); none receives a broad or supercluster label, and 54% / 88% keep only a confident neuronal lineage. The excess is .02 / .04 points; we propose to accept it for these two datasets." (The vocab option was named by §7 item 1; it and the metric option are not adopted.)
- **D7, H4 on HD2_MERSCOPE** (`CRITERIA_AFTER_M4.txt` §2.3, corrected by C6 / C7 for the WHB-only headline set with the drop): "On [HD2]_MERSCOPE (median 17 counts, broad_only) H4 passes 4 of 7 classes on the WHB-only held-out label set. Every class shows fold ≥ 5.8; the three failing classes (Oligodendrocytes, Microglia, Vascular cells) reach the maximum AUROC their held-out-marker detection permits (AUROC .646 / .540 / .663 against ceilings .654 / .541 / .669), so the failure measures marker detection on this shallow section, not labelling error; on cells ≥ 30 counts 6 of 7 pass (Microglia .565 at its ceiling .566, measured with the pre-registered test set)." M4's text claimed "fold ≥ 6" and gave the SEA-voted set's ceilings; both are corrected here.

**Not covered by these decisions** (they come back to the user with the M8 numbers before the gate PR): the mouse rows of H18 (CHECK M2: MS1 PREP subclass Immune at 100 counts, which passes with seeds 1 and 2; MS1 reweighted at 250 counts; would-raise MS1 39, MS2 68), whether H18's emission test covers the SEA-AD bundle (CHECK M3: its version-6 PREP withholds broad Astro 120, Fibroblast 15 and Oligo 120 before the drop), M7's D23 (tangential-block CI for H12, §17), applying D1 to human families beyond the M8 set a / set c / SEA-AD bundles (5K, the M13 custom panels, gate P's leave-one-donor-out sets: the set a evidence behind D1, PREP broad Oligo precision, does not carry over to other panels, and the drop would change the M13 acceptance test set and remove supercluster COP from H18 there; if the user declines, `HO_SELF_MAP_TEST_SET_REVISION` is keyed to the set a panel families only), and any failure the M8 run finds outside the texts above.

**D6 as implemented.** H4's scored value is the `H4` row of `scripts/acceptance/resolve_criteria.py`, which is the WHB-only re-resolve of the held-out WHB re-map (`m4_resolve_heldout_whb_only`: degraded mode `whb_only`, no method saw the held-out genes); every other label set (the SEA-voted `m4_resolve_heldout`, the circular production set, M3's (a)–(c) of `heldout_genes.py`) is reported as an `H4[<set>]` row that is not scored.

**D9, confirmed.** The rule is implemented in `consensus.resolve_human` (a neuron's supercluster parent is its NT: `leaf_parent`) and tested (`test_nt_is_not_applicable_to_non_neurons_and_gates_neuron_leaves`); nothing changes.

**M8 scoring protocol** (fixed here, before the M8 run):

1. **Order.** D1 is implemented and the human self-maps rebuilt into new store build directories before any M8 criterion is measured; no existing bundle is overwritten.
2. **Draw.** Every resolvability-dependent verdict (H7, H8, H18, and through RESOLVE every label) uses the pre-registered seed-0 draw (per-cell seed 0, efficiency seed 0, mapping seed 0), mapped with the self-map's recorded MapMyCells worker count (8, `mapping_runs` of `bundle.json` / `resolvability_summary.json`; ctm seeds each chunk of ceil(n / workers) rows, so another count is another realisation under the same build hash). Before scoring, stage B checks that every PREP bundle it used was reused (not rebuilt) or records 8 workers; otherwise the result goes back to the user instead of being scored. (Clarified at the M8 review, 2026-09-30, before stage B: it names the draw stage A2 scored, the stored 8-worker rows, and moves no verdict.) The draw-spread table is an acceptance output beside it: the 3 × 3 grid of per-cell seeds 0–2 × efficiency seeds 0–2 on the rebuilt set a self-map, each draw decided alone with the unchanged rules (unweighted for PREP, reweighted to each dataset for RESOLVE), with, per draw and dataset, H7, the H8 warning value, H18's failing pairs and the validated bins emitted, and per dataset and metric the seed-0 value, the min–max over the 9 draws and the number of draws failing (`draw_spread.csv`, `merxen.annotation.draw_spread`). It never changes a verdict. The vote, pooled and v5 rules (memo R2–R7) are not adopted; any of them would need the user's written approval.
3. **H12.** Scored on the square 500 µm tile CI (the `depth_ordering_passes` records with `kind = square_tile_500um` in `acceptance_metrics.json`, per platform, AND over both platforms, cross-checked with the pair's `depth_ordering_replicated` record of that kind; and the tile CI of the WM − GM contrast, its lower bound above 0 on both platforms); the tangential-block CI, the report's primary unit since M7, is reported beside it (the kind-less records). `report_depth.SCORED_CI` and the `depth_ci_scored` record name the scored CI, and `report_scoring.score_h12` scores it and records `ci_scored` for `summary.json`. The user has not approved scoring on tangential blocks (§17, M7 D23). MERSCOPE H12 stays `not_available` until the native-frame depth re-run is in (§17).
4. **First measurements.** H6, H12, H13, H14 and H15 are measured for the first time at M8. No exception is pre-approved for them: any failure comes back to the user before the gate PR.
5. **H4.** The WHB-only headline set is scored (D6); the D7 exception covers HD2_MERSCOPE only.
6. **Exceptions.** D3, D4, D5 and D7 cover the datasets, bins and classes they name. A failure outside that scope, or one that the M8 run finds materially different from the text, comes back to the user; the gate PR quotes the texts with the M8 numbers. "Materially different" is checked, not judged by hand (`report_scoring.check_d3` … `check_d7`; tolerances stated at the M8 review, 2026-09-30, before stage B; they only narrow the exceptions): **D3** the gate level is `broad_only` and the warning value lies within .148 ± .005; **D4** broad OPC misses at most 15 and 120 counts on the eight datasets (HD3_XENIUM 15 only), broad and supercluster Immune at most 60 counts on HD4_MERSCOPE only, PREP's would-raise list is within the text's 15 bins (equal at stage A2) and each dataset's within the stage A2 list (`$A/m8/stageA3/d4_would_raise_reference.csv`), every added bin itemised; **D5** H2 exceeds the text's value (1.02% / 1.04%) by at most .05 points and no flagged cell carries a confident broad or supercluster label; **D7** the failing classes are among Oligodendrocytes, Microglia and Vascular cells, each with fold ≥ 3 and AUROC within .01 of its detection ceiling, and ≥ 4 classes pass. A row inside its text is `EXCEPTION (Dn)`; outside it `EXCEPTION-RECHECK (Dn)`, which goes back to the user; a failure no decision names is `FAIL-OUTSIDE`.
7. **Held-out results** (HD3, HD4, reseg) are reported separately and select nothing (rule 3).

**Stage A2 re-measurement (2026-09-30; recorded, no decision).** D1 was implemented and the three human self-map bundles rebuilt into new store build directories (set a WHB `ffc6dd2d`, set a SEA-AD `dc3c500f`, set c WHB `2ed1c2f6`; 132 of 11,040 test cells left out; 2.1–3.3 min and ≤ 2.9 GiB peak RSS per build (≤ 9.9 GiB summed over the process tree); `$A/m8/stageA2/`). The rebuilt set a WHB self-map equals M8 prep's `resimdrop` tables in every row. RESOLVE was re-run standalone on the M3 shadow MAP outputs of the 8 datasets × proseg_hybrid and reseg (`$A/m8/resolve/`; their calls equal the published MAP outputs where both exist), and every criterion M4 re-measured was re-scored with this protocol (`$A/m8/CRITERIA_M8_STAGEA.txt`). Every value equals the M8-prep projection (H7 HD4_MERSCOPE .436, the HD2_MERSCOPE warning .148, set a PREP passes H18, 10 reweighted H18 pairs, broad coverage within ±0.05 of M3 on all 8 proseg_hybrid datasets, worst −.023). The scored failures are exactly the ones D3, D4, D5 and D7 name, with the numbers of their texts; none falls outside them. The rebuilt SEA-AD self-map also withholds broad Vascular at 10 counts (not in CHECK S1, which was table-level); RESOLVE reads only the primary bundle's resolvability tables, so no label changes. Over the 9 draws (D2) HD4_MERSCOPE's H7 is .430–.451 and HD2_MERSCOPE warns in 5; the reweighted H18 pairs are 0–13 (the memo's 0–14 was the table-level drop). MapMyCells output depends on the worker count (ctm seeds each chunk of ceil(n / workers) rows), so the draws are mapped with the self-map's 8 workers; the build hash does not record the count (it is in the bundle's `mapping_runs`). H6 and H12–H15 are not measured until the pipeline runs (stage B).

**M8 review (2026-09-30; recorded before stage B, no decision).** The review of stage A2 found that H12's scored CI, the exceptions' scopes and the scored draw's worker count were written here but not encoded. Since then `report_depth` writes the square-tile ordering records (per platform always, and the pair's `depth_ordering_replicated` with `kind = square_tile_500um`) and `depth_ci_scored`, and `report_scoring.score_h12` scores them (item 3); `report_scoring` checks the D3 / D4 / D5 / D7 scopes with the tolerances of item 6; item 2 names the 8-worker draw; H18's floors are the bundle's panel family's (only `human_set_a` is packaged, so no value moves); the report's item 8 records only the assigned H4 set under H4; the bundle lookup reads the test-set revision from the hashed build params. Re-scored on stage A2's outputs (`$A/m8/stageA3/`, `$A/m8/CRITERIA_M8_STAGEA.txt` §8): the criteria table, the scored draw's H18 rows, bins and would-raise counts equal stage A2's; the scored draw re-mapped with 8 workers reproduces the stored rows; every scored failure is inside its text (D3 1, D4 20, D5 2, D7 1 rows), none needs a recheck, none falls outside. *Diagnostic only* (H12 is first measured in stage B): on the M7 reports with the native-frame depth (`$A/m7/reports_depthfix/`), HD1's ordering fails on the scored square tiles (Xenium) and passes on the tangential blocks beside it; WM > GM passes on both platforms of HD1 and HD2. Unless the user approves the tangential blocks (M7 D23) or an exception, HD1's H12 is expected to fail in stage B and comes back to the user (item 4).

**Stage B scoring as implemented (recorded 2026-09-30, before stage B; no decision).** `scripts/acceptance/run_acceptance.py score` applies items 1–7 to the stage B outputs; §9 fixes the first-measured criteria's thresholds but not their operational definitions, so these are stated here before any stage B measurement and come back to the user with the results:

- **Before scoring (item 2):** every PREP bundle the run used must be one of the three D1 bundles and have been reused (PREP's log, or `bundle.json` created before the run) or its self-map must record 8 workers, and every MAP run must name one of them; otherwise every row is `UNSCORED (P5)`.
- **H1–H3, H5, H7–H10, H16, H17:** the rows of `resolve_criteria.py` on the pipeline's RESOLVE outputs, scored as in stage A3; each value is cross-checked with the pair × segmentation's `acceptance_metrics.json` (and H10 with `marker_referee.py`), and a difference beyond the report's 6-significant-digit rounding goes back to the user.
- **H6:** the E2 in-silico cells of the `contam` scenario with the held-out-donor WHB engine; confident at the v1 raw thresholds (broad bp ≥ 0.73; supercluster bp ≥ 0.69 with a confident broad), each cell weighted by the dataset's share of its truth class over the simulation's share at that depth (E2's reweighting), the dataset's composition being the §5.5 soft mass of its proseg_hybrid table cells on the WHB superclusters (the mass the production reweighting uses), aggregated to E2's broad classes through the simulation's truth mapping; per depth (broad D ≥ 15, supercluster D ≥ 30) and class with ≥ 50 test cells, the reweighted precision of the confident calls of that class ≥ 0.90 / 0.85, on each of the 8 proseg_hybrid datasets. The other reading of "classes with n ≥ 50", the precision of all those classes' confident calls pooled per level and depth, is reported beside it (`H6/pooled`) and not scored; which reading H6 means is the user's to confirm. Two points of that confirmation (added at the stage B review, 2026-09-30, before any stage B measurement): (a) the stage A2 projection on the E2 cells (`$A/m8/stageB/preflight/smoke_stage_a_inputs/`) already shows the per-class reading failing and the pooled reading passing on all 8 datasets, so adopting the pooled reading now is a post-hoc loosening under rule 2 (like the H12 CI in §17) and needs the user's written approval in the gate PR; (b) the flip rule says "H6 and H18 once" and rule 3 keeps the held-out donors separate, while this definition scores H6 on the compositions of all 8 datasets, the held-out HD3 and HD4 included; whether H6 is scored on the held-out donors' compositions or only reported for them is the user's to decide.
- **H13:** MENDER status `complete` on both platforms of every pair under both policies (`exclude_from_features` in the main run, `state` in the second-policy run); cortical-depth violins published for every segmentation and platform; the report's `table_ids_equal_clustered` true on every segmentation and platform; every table group, legacy clustered output and legacy MENDER output that existed before the run unchanged by sha256 (`compare_legacy.py untouched`); the viewer on reseg is the user's manual check.
- **H14:** per pair, the realtime of ANNOTATE_PANEL + MAP + RESOLVE summed over its four segmentations ≤ 2.5 h and the largest MAP peak RSS ≤ 16 GB (Nextflow trace); per PREP bundle, its recorded build time incl. resolvability ≤ 2.5 h; no GPU process in the trace and the GPUs hidden in every task.
- **H15:** on HD1 and HD2 proseg_hybrid, the pipeline's MAP and RESOLVE tasks re-run in fresh directories with the same staged inputs and command: every MAP and RESOLVE parquet identical in content; and MAP re-run with `rng_seed` 1 (every reference) then RESOLVE: the cells whose confident broad label differs (confident under one seed only, or under both with another name) ≤ 1% of the seed-0 confident broad labels.

## 19. M8b segmentation comparison, reseg vs proseg_hybrid (requested and approved by the user; pre-registered 2026-09-30, before any M8b analysis)

Requested and approved by the user in the chat on 2026-09-30 (plan §12 M8b). This section fixes the decision question, the arms and the metrics before any M8b analysis is run. **No threshold, target or decision rule of §9 changes, and M8b scores nothing for gate H:** the M8 verdicts are those of §18. **No threshold is fitted after looking at the results:** the user decides OD-B7′ from the evidence. OD-B7 (proseg_hybrid for human) stays in force until then. *To confirm (stage B review, 2026-09-30):* the user approved the M8b plan draft (plan §12 M8b); the operational choices below that the draft does not state were set when this section was written, and the user confirms them before M8b runs: the thinning of `hybrid_matched` (multivariate hypergeometric without replacement, seed 0, keyed by cell id; mapped with 6 workers), analysis 2's tests (Spearman ρ with a 1,000-resample gene bootstrap, Mann–Whitney, Wilcoxon signed-rank, Benjamini–Hochberg over the analysis), analysis 3's distance bins (0–10, 10–15, 15–20, 20–30, 30–50, > 50 µm) with a 200-resample 500 µm tile CI, and analysis 3's per-gene metric (the detection rate of a class's negative genes in that class, for the draft's "cell-type-exclusive genes detected in cells of other types"). *Confirmed (2026-10-01, before any M8b analysis):* the user confirmed every operational choice above as written ("Confirm all") and chose the output location `results/acceptance/2026-09-30/segmentation_comparison/` (the M8 runs' acceptance date) for `results/acceptance/<date>/segmentation_comparison/` below. The M8b scripts write only under `$A/m8b/`; the orchestrator copies the final outputs to that location.

**Decision question.** Which segmentation should be the human default: (a) for cell typing; (b) for per-gene expression within cell types; (c) for both at once, via a combined mode: type cells on reseg counts, and measure expression on proseg_hybrid counts of the same cell ids.

**Datasets and arms.** HD1, HD2, HD3, HD4 × MERSCOPE, Xenium (8 datasets; HD3 and HD4 reported separately, rule 3). Arms, all from the M8 stage B outputs:

- `reseg`: the reseg map_first outputs (table `table_MOSAIK_proseg`: ProSeg's `assignment`);
- `hybrid`: the proseg_hybrid map_first outputs (`table_MOSAIK_proseg_hybrid`);
- `hybrid_matched`: the proseg_hybrid table cells whose id is also a reseg table cell, each thinned to the reseg cell's total count by a draw without replacement from its own counts (multivariate hypergeometric, seed 0, keyed by cell id; a cell whose proseg_hybrid total is at most its reseg total keeps its counts), then mapped with the production MAP configuration (the D1 bundles, 6 MapMyCells workers, seed 0) and resolved with production RESOLVE. It removes the depth difference, so a difference that remains between `reseg` and `hybrid_matched` is not a count-depth effect;
- `combined`: the proseg_hybrid table cells carrying the RESOLVE labels of the reseg cell with the same id (cells without a reseg table cell are unlabelled), with proseg_hybrid counts;
- context: the proseg_mask and original_seg stage B outputs, reported beside them only.

**Analysis 1: side-by-side scores.** Every record of each arm's `acceptance_metrics.json` (the M7 report; for `hybrid_matched` a report built on its MAP / RESOLVE outputs, all other inputs unchanged) and every §14 criterion value of the acceptance scripts (`resolve_criteria.py`, `marker_referee.py`, H12 by `report_scoring.score_h12` on the scored square-tile CI with the tangential blocks beside it, on the corrected depth), computed identically for each arm. Per dataset and platform: each arm's value, the differences `reseg − hybrid` and `reseg − hybrid_matched`, and the metric's own 95% CI (a paired CI where the metric's bootstrap supports one, e.g. `paired_block_bootstrap_jsd_difference` for the JSDs; otherwise the two CIs side by side). Coverage and annotatability are also reported per depth bin of the bundle's grid (10, 15, 30, 60, 120, 250 counts) and level. The items: annotatability and coverage by depth bin, composition and cross-platform JSD, the marker referee, negative-marker contamination, reference-expectation pseudobulk r, method agreement (WHB–SEA-AD), held-out genes (H4, the WHB-only headline set), cortical-depth ordering, OOD / diffuse flags and the dataset gate. No metric is added or dropped after the first results; a metric an arm cannot provide is reported as `not_available` with the reason.

**Analysis 2: transcript-loss audit (reseg).**

- *Transcripts:* the rows of `points/transcripts` that pass the decoding filter the segmentations used (qv ≥ the samplesheet's `xenium_min_qv`, 20, on Xenium; on MERSCOPE the filter recorded for the ProSeg run, none if none), control features excluded (plan §8.4 rules). *Assigned under reseg:* `assignment` set and `background` false, to a reseg table cell; *under proseg_hybrid:* `hybrid_assignment` set and `hybrid_background` false, to a proseg_hybrid table cell. *In a nucleus:* the transcript's native-frame (x, y) lies inside a `shapes/cellpose_nuclei` polygon.
- *Per gene g and platform:* A_reseg(g) and A_hybrid(g), the assigned shares of the decoded transcripts; N_bg(g), the share of in-nucleus transcripts that reseg leaves unassigned; the loss L(g) = 1 − A_reseg(g) / A_hybrid(g). *Per gene within each confident cell type c* (broad; supercluster where the dataset gate is `full`): the same shares over the transcripts of the proseg_hybrid cells whose confident proseg_hybrid label is c (an in-nucleus transcript belongs to the proseg_hybrid cell that holds that nucleus).
- *Covariates:* expression level (log10 of the mean proseg_hybrid counts per table cell); gene length (the gene's longest annotated transcript, from a local annotation table if the host has one; otherwise reported as not available); probe count (from the panel or codebook file where available; otherwise not available); membership of a nuclear-retained or process-localised gene list, fixed in the M8b script with its literature source before the script's first run (genes not on the panel dropped); platform; cell class (glia: Astro, Oligo, OPC, Immune, Vascular; neurons: Exc, Inh).
- *Tests:* per platform, Spearman ρ of L(g) and of N_bg(g) with each continuous covariate, with a 95% bootstrap CI over genes (1,000 resamples, seed 0); a Mann–Whitney test for the gene-list membership; glia vs neurons as the per-gene difference of L between glial and neuronal cells (Wilcoxon signed-rank over genes); platforms paired over genes. Benjamini–Hochberg over this analysis's tests; every p value is reported with its effect size.
- *Effect on expression:* per dataset, confident broad class c and gene g, the pseudobulk share p_arm(g, c) = (Σ counts of g + 1) / (Σ counts of the panel genes + n_genes) over the cells confident c in both arms (same ids); the log2 ratio log2(p_reseg / p_hybrid); the per-gene factor against the WHB reference, log2(p_arm(g, c) / p_WHB(g, c)) with p_WHB from the primary bundle's class profiles aggregated to the broad class, reported per arm with its spread (q90 − q10 over genes) and the Spearman ρ of the two arms' factors. *Marker contrasts:* for each canonical marker g of class c1 (the E1 referee's list) and each other class c2, log2(p_arm(g, c1) / p_arm(g, c2)) per arm and the difference reseg − hybrid.

**Analysis 3: spill-over audit (proseg_hybrid).** On the cells confident at broad level in both `reseg` and `hybrid` (same ids): the negative-marker `contamination_score` RESOLVE writes (counts on the assigned class's negative genes, bundle `negative_genes.parquet`, over total counts), and per negative gene of class c the detection rate (share of the class-c cells with ≥ 1 count of it), both as a function of the distance from the cell's centroid to the nearest centroid of a confident cell of another broad class (native frame; bins 0–10, 10–15, 15–20, 20–30, 30–50 and > 50 µm), per class and arm. Reported: the hybrid − reseg difference per bin with a 95% CI from 200 resamples of the §5.5 500 µm tiles (seed 0), and per gene the hybrid − reseg difference of its detection rate in the classes where it is negative.

**Analysis 4: combined mode (c).** On `combined`: the share of proseg_hybrid table cells that get a confident reseg label per level, and the share without a reseg table cell; the confident composition and the cross-platform confident broad JSD (whole section, §2's matched-tile CI) beside those of `reseg` and `hybrid`; the analysis-2 expression metrics with the combined labels on proseg_hybrid counts. The pipeline change the mode would need is stated; it is not implemented in M8b.

**How the evidence maps to the question.** (a) typing: analysis 1, overall and at matched counts; (b) expression: analyses 2 and 3; (c) the combined mode: analysis 4 against (a) and (b). The decision memo gives each option with its measured effects, a recommendation that names the measurements it rests on, and the side effects; a metric that favours different segmentations on different datasets or platforms is reported as such, not averaged away. Outputs: `results/acceptance/<date>/segmentation_comparison/` (HTML and one CSV per analysis) and the memo; the user decides OD-B7′, and the flip PR follows that decision.

## 20. Decisions of 2026-10-01 (M8 stage B follow-up; recorded before stage B is re-scored)

Stage B (`results/acceptance/2026-09-30/summary.json`, 2026-10-01 02:30) sent 15 rows back to the user: H6 on the 8 proseg_hybrid datasets, H12 on HD1 proseg_hybrid (and, as information, HD3 reseg and proseg_mask), H15/seed1 on HD1 and HD2, H17/H2 on HD2_MERSCOPE reseg (`EXCEPTION-RECHECK (D5)`) and the H13 viewer check. The orchestrator put nine questions in the chat (H6 a / b / c; H12 a / b; H15 a / b / c; D5 extended; the viewer check; §18's definitions; the flip PR's controls 1 / 2 / 3; M3c; PR #32). **The user's answers, given in the chat on 2026-10-01:** "1. Choose b 2. Choose b 3. Choose A and B 4. Yes 5. [the viewer check, by the user] 6. Looks good 7. 1 8. accept it as is and report the stability result 9. The hitl plan looks good for now." No threshold or target of §9 changes.

| # | Question | Decision | Label (§1 rules 1–4) |
|---|---|---|---|
| D10 | H6 (per-class simulated precision) fails for OPC, COP and deep-layer CT / 6b; the pooled reading passes. (a) adopt the pooled reading / (b) a written exception for those classes at those depths / (c) stop emitting supercluster OPC below about 60 counts on COP-rich datasets | (b) | exception; the per-class reading stays the scored one, `H6/pooled` stays reported beside it |
| D11 | H12 on HD1 fails on the square tiles (Xenium ordering) and passes on the tangential blocks. (a) a written exception / (b) score the tangential-block CI | (b) | loosening (rule 2), approved in writing; supersedes the square-tile choice of M7 D23 (§17) and §18 item 3 |
| D12 | H15/seed1: 7.0% (HD1) and 13.4% (HD2) of the seed-0 confident broad labels change with MapMyCells `rng_seed` 1, and none is a switch between two confident names. (a) score only true label switches / (b) a written exception recording the threshold noise / (c) 1,000 bootstrap iterations | (a) and (b) | (a) is a reinterpretation of a first-measured criterion, treated like a loosening (rule 2) and approved in writing; (b) is an exception |
| D13 | D5's text gives HD2_MERSCOPE reseg H2 1.04%; stage B measures 1.13%. Extend D5 to the pipeline's value? | yes | exception (D5's scope unchanged, its number updated) |
| D14 | §18's operational definitions of the first-measured criteria (H6, H13–H15) | confirmed as written | definition; this includes scoring H6 on the compositions of all 8 datasets, the held-out HD3 and HD4 included (§18, H6 point b) |
| D15 | The flip PR's controls: (1) a per-row `clustering_squidpy_mode` samplesheet column / (2) map-first for chosen segmentations only / (3) keep the legacy tables by default after the flip | (1) only | pipeline design (plan §12 M8, flip PR) |
| D16 | M3c after its stability test failed twice (`$A/m3c/M3C_EXIT_REPORT.txt` §10) | accept version 7 as is and report the stability result (test iii) instead of a pass / fail | loosening (rule 2), approved in writing; it also confirms M3c's D1–D7 |

PR #32 (the Cellpose human-in-the-loop plan) is fine for now and stays open; HITL-0 pins cellpose 4.2.1.1 and `cpsam_v2` later (user, 2026-09-30).

**Approved exception texts** (the gate PR quotes them with the re-scored numbers and keeps the scope):

- **D10, H6** (stage B values): "On the in-silico `contam` cells reweighted to each human dataset, the per-class precision of confident calls misses its target for the oligodendrocyte-lineage precursors and one deep-layer neuron supercluster only: broad OPC at 15 counts on all 8 datasets (.689–.893), at 30 counts on 5 (.811–.889), at 60 counts on [HD3]_MERSCOPE (.883) and at 120 counts on 5 (.799–.885); supercluster Oligodendrocyte precursor at 30, 60 and 120 counts on all 8 (.236–.639, .263–.673, .367–.769); supercluster Committed oligodendrocyte precursor at 30 counts on 8 (.611–.850), 60 on 5 (.705–.803) and 120 on 8 (.589–.841); supercluster Deep-layer corticothalamic and 6b at 30 counts on 5 (.768–.845). The pooled precision of each level and depth passes on all 8 datasets (lowest .944). Broad OPC at 15 and 120 counts is also where resolvability withholds the label on the real datasets (D4)." **Scope:** the 70 failing (dataset, level, class, depth) rows of stage B's `tables/h6_precision.csv` on proseg_hybrid, i.e. the classes, levels and depths named, on the datasets that fail them there; each row's precision may fall at most .01 below its stage B value (a tightening stated at the review, 2026-10-01, before the re-score; `report_scoring.D10_FAILING`, `D10_TOLERANCE`). *Corrected 2026-10-01, before the re-score (orchestrator):* the text first recorded in this section (commit ec49874) counted the failing datasets from `summary.json`'s per-dataset notes, which list only the first 8 failing rows, and so read supercluster Oligodendrocyte precursor "60 on 7 … 120 on 3" and Committed oligodendrocyte precursor "120 on 4 (.643–.841)". The full table gives all 8 for each (COP 120: .589–.841). The classes, levels and depths are those of the question the user answered ("those classes at those depths"); only the dataset counts changed, and the gate PR shows the user the corrected list. It also covers supercluster Fibroblast at 120 counts on HD4_XENIUM (.783), which stage B fails too; the user extended the exception to it on 2026-10-01 ("Except this too").
- **D12 (b), H15/seed1:** "Re-running MAP with MapMyCells `rng_seed` 1 changes 7.0% ([HD1]) and 13.4% ([HD2]) of the seed-0 confident broad labels. None is a switch between two confident labels: every change is a cell whose bootstrap probability crosses the confidence threshold (confident under one seed only). With 100 bootstrap iterations the probability has a standard error of about 0.045 near the thresholds (0.69–0.73), so cells near them flip; raising it to about 1,000 iterations (option c) would make MAP about 10× slower and was not adopted." Under D12 (a) the scored H15/seed1 value is the share of seed-0 confident broad labels that are confident under seed 1 with another name (≤ 1%). The threshold-crossing share stays a scored row of its own (H15/seed1_threshold, ≤ 1%), which D12 (b) covers. **Scope of D12 (b):** HD1 and HD2 proseg_hybrid; no switch; no cell missing from either seed's table (a missing cell would count as a crossing); and the crossing share at most the text's value (7.0%, 13.4%) plus .05 points, D5's tolerance (stated at the review, 2026-10-01, before the re-score; `report_scoring.check_d12`).
- **D13, D5:** D5's text reads "[HD2]_MERSCOPE reseg (1.13%)" in place of 1.04%, measured by the pipeline (stage B); its tolerance (at most .05 points above the text) and its condition (no flagged cell carries a confident broad or supercluster label) are unchanged.

**D11 as encoded.** The scored H12 ordering CI becomes the tangential-block CI (the kind-less `depth_ordering_passes` records and the pair's kind-less `depth_ordering_replicated`; a platform without tangential positions keeps the square tiles as its only CI, as in the report); the square-tile ordering records are reported beside it. WM − GM keeps its square-tile CI, because white matter has no tangential position (the only CI the report computes for it). `report_depth.SCORED_CI`, the reports' `depth_ci_scored` record and `report_scoring.score_h12` follow.

**Re-scoring.** Stage B's outputs are re-scored with D10–D13, unchanged otherwise; nothing is re-run. Stage B's `summary.json` stays as it is. The re-scored rows are the gate PR's numbers, and any row the re-score finds outside these texts comes back to the user.

**Re-score (2026-10-01, code 2629065, verdicts identical to the first re-score with 4ae2c1e; `results/acceptance/2026-09-30/rescore_2026-10-01/`, script and evidence `$A/m8/rescore_20261001/`).** Read-only on stage B's outputs: the H15 compare re-read stage B's re-run and seed-1 outputs to add the switch count, and `score` re-read everything else. Exactly 13 rows change, all as D10–D13 intend: H6 on the 8 datasets `FAIL-OUTSIDE` → `EXCEPTION (D10)` (every failing row inside D10's list); H12 HD1 proseg_hybrid `FAIL-OUTSIDE` → `PASS` (and HD3 proseg_mask `INFO fail` → `INFO pass`); H15/seed1 on HD1 and HD2 `FAIL-OUTSIDE` → `PASS` (0 switches of 192,005 and 123,535 seed-0 confident broad labels), with the new rows H15/seed1_threshold `EXCEPTION (D12)` (7.02%, 13.40%); H17/H2 HD2_MERSCOPE reseg `EXCEPTION-RECHECK (D5)` → `EXCEPTION (D5)`. Every other row is identical in value and verdict. Two rows still go back to the user: the H13 viewer check (the user's manual check), and H12 on HD3 reseg (`INFO fail`: not scored, a held-out pair and not the default segmentation; its MERSCOPE ordering fails on the tangential blocks too, WM > GM passes on both platforms; item 4 sends every first-measured failure back).

**OD-B7′ and the M8b follow-up (user, 2026-10-01, after reading the M8b memo).** The user's answers to the three questions put after M8b: "1. Lets go with A, we'll still get the reseg outputs right? So we can always do supplemental analysis on [HD1] manually with reseg if we like 2. Except this too 3. Leave as they are".

- **OD-B7′ = A:** proseg_hybrid stays the default human segmentation for cell typing and for per-gene expression (OD-B7 confirmed). The combined mode is not built. Map-first typing still runs on every segmentation, reseg included, so reseg's labels stay available for supplementary analysis (for example on HD1); the default segmentation sets MENDER's default (`mender_segmentations`, overridable per samplesheet row), the cross-platform sensitivity check and the gate's scored segmentation. Stage B's scored segmentation is unchanged, so D3, D5, D7 and D10 keep their texts.
- **D10 scope:** extended to supercluster Fibroblast at 120 counts on HD4_XENIUM (above).
- **`hybrid_matched`'s count-based items** (H9, H10 and report items 4 and 7 read the unthinned counts; `tables/posthoc_matched_count_sources.csv`): left as they are and labelled; no decision rests on them.
- The M8b post-hoc tables (labelled as not pre-registered) are published with the comparison; they fit, score and select nothing.

## 21. M3c acceptance tests (pre-registered 2026-09-28, before any M3c code)

*Numbering:* this section and §22 were written as §14 and §15 on the M3c branch and renumbered when M3c merged after M8 (2026-10-01); `$A/m3c/` evidence files written before then (e.g. `M3C_EXIT_REPORT.txt`) cite them as §14 and §15.

Milestone M3c ("Xenium Prime 5K measured efficiency, per-class depth and emission stability": plan §12 "M3c", §8.3 "Resolvability version 7" items v7.1–v7.12, §8.10) is pre-registered here on 2026-09-28, on `feature/rca-m3c-5k-measured-efficiency` at `83e81e3` plus documentation commits only: no M3c code exists yet. The rules of §1 apply: nothing below may be loosened without your written approval in the M3c PR, and every tolerance and constant may only be tightened. The evidence is the phase-1 5K real-data programme (`$A/5k_real/SYNTHESIS.txt` and the reports it cites); the numbers behind the tolerances are in `$A/m3c/STAGE_A_DESIGN_NUMBERS.txt`. Every M3c run keeps to ≤ 8 processes and ≤ 64 GB per job, at most two heavy jobs at once, CPU only.

### 21.1 Decisions of 2026-09-28 recorded with this pre-registration

Your decisions (plan decision register D-G9–D-G14; open decisions OD-E1, OD-E2, OD-E13–OD-E17):
1. **OD-E1 amended.** Public vendor data may supply simulation inputs (per-gene factor tables, per-class depth profiles), stored with provenance (source URL, sha256, licence note, XOA / segmentation version, deriving script), but never promote trust; real data stay downgrade-only.
2. **OD-E2 / M10 split.** Simulation-side measured factors move into v1; applying platform factors to real counts stays v1.1.
3. **R3_measured_HO** is one member of the draw ensemble in both PREP and gate P, not the base recipe.
4. **Glial optimism.** A panel-card note ("simulated glial coverage is an upper bound: −.06 to −.18 on vendor-segmented 5K cells"; "precision unmeasured on real data") and a downgrade-only per-class real-vs-simulated coverage warning (real < simulated − 0.10); no empirical offset.
5. **Thin classes** (fewer than 50 confident PREP calls, e.g. DG-IMN and MH-LH on 5K mouse): the PREP test set is topped up from reference cells not used by the marker build, respecting the donor hold-out and the training-cluster rule; verdicts are never borrowed from real data.
6. **Downloads.** Only the 5K mouse `transcripts.parquet` (2.30 GB) is approved; not v1 or lung transcripts, morphology images, outs bundles, an Ensembl GTF or anything else.
7. **Neighbour-structured spill** stays experimental, behind a flag, off by default.

The M3c scope decision (protecting this pre-registration and the in-flight M4 branch; D-G11, OD-E12): the resolvability version-6 path (R1_contam_HO, the pre-registered seed-0 draw, the current thinning and depth grids) stays byte-identical and stays the emission basis of the families whose decisions are pre-registered for gates H and M, the families of `validated_panels.csv` (human_set_a, mouse_ms1, mouse_ms2); version 7 is selected per family and is the default for every other family (Xenium Prime 5K mouse and human, unknown and custom panels, including M13's custom MERSCOPE panel). For the version-6 families the version-7 decisions and the ensemble churn are computed as a diagnostic only, reported and not applied, for the M8 decision. RESOLVE's use of each dataset's own per-class depth needs M4's RESOLVE and is a follow-up (plan §12 M3c). Two items arise from applying it (plan OD-E18–OD-E20): HD4 MERSCOPE's own 268-gene family, a gate-H dataset outside `validated_panels.csv`, is pinned to version 6 (OD-E20, recommended default adopted unless you object); NP9's wall-time reference for version-7 families is open with no default, because every option but keeping it loosens a criterion (OD-E18).

None of these changes a threshold of §9. Plan §14 gained one paragraph, "Version-7 families", which adds conditions to gate P and removes none (a tightening); verbatim:

> **Version-7 families** (M3c; recorded 2026-09-28, before any version-7 family is scored; §8.3 v7, §8.10). Gate P tests what production emits: the thresholds and emission it freezes are those of the version-7 ensemble (§8.3 v7.7–v7.9), derived once from the default donor or draw. NP3–NP7 are then scored separately in **every emission member** — each member's simulation of the held-out calls at the frozen thresholds; for NP4 each member's own replicates over donors or draws and seeds; NP6's stress recipes applied to each R1 member as defined above and to R3 as spill 0.35 only — and a (level, class) is validated only if it passes in every member. NP3's base recipe stays R1_contam_HO at seed 0 (R1@0, a member); R3 is never the base (user decision 3). Human Prime families add `R1_xtissue_lung_stress` to NP6's stress recipes. NP5's expected depth is per class, the class's median in the family's per-class depth profile where one exists (§8.3 v7.5). The tested-set rules apply unchanged, so a bin validated by gate P passes on its tested set whatever route emitted it (monotone-filled bins included). The public 5K section is never gate-P evidence. These conditions are added to the rules above and none is removed, so they tighten the criteria (§21 Rules); NP9's wall-time reference for version-7 families is open (OD-E18).

### 21.2 Tests

Tests (i), (ii), (iii), (v) and (vi) are pass / fail; (iv) and (vii) are reported and have no pass / fail outcome.

**(i) Version-6 byte identity for the pre-registered families** (pass / fail). Scope: every family that resolves to version 6 (plan §8.3 v7.1) — human_set_a (set a 296 / 297 and set c 264 / 265), mouse_ms1, mouse_ms2 and the pinned HD4 MERSCOPE family.
- **(i-a) Golden hashes in CI** (`tests/test_annotation/test_resolvability_v6_golden.py`, fast). On the synthetic references of `tests/test_annotation/conftest.py` with the deterministic fake mapper of `test_resolvability.py`, one human-like and one mouse-like synthetic panel run the whole version-6 path: test cells, `thin_and_contaminate` for R1_contam_HO and clean, mapping, `level_cells`, `decide`, floors, trust constraint, `resolvability_table` and `build_summary`. Hashed (sha256): (1) each recipe's simulated query — the CSR `data` (float64), `indices` (int32) and `indptr` (int64) as little-endian bytes, and the obs index; (2) the decisions of every regime as canonical CSV (rows sorted by regime, level, class and depth; columns in `decide`'s order; floats as `%.17g`, missing values empty); (3) the `resolvability.parquet` table as canonical CSV (rows sorted by every key column); (4) the summary JSON without `runtime_s`, keys sorted; (5) `compute_build_hash` of a version-6 bundle spec for each seeded panel hash of `validated_panels.csv`. The golden values are computed with the code of `83e81e3` (a `git archive` export in the scratch directory) before any behavioural change and committed with the test, which must then pass unchanged at every later M3c commit. The version-selection truth table (listed, inherited, pinned and unknown families) is tested beside it.
- **(i-b) Stored production bundles** (`scripts/acceptance/m3c_v6_identity.py`; no mapping). The version-6 bundles in the SSD1 store on 2026-09-28 — WHB set a 297 `b6bfe83d…`, WHB set c 265 `5137090d…`, SEA-AD `d70cda25…`, WMB MS1 `5a032858…` and WMB MS2 `daa8c4a6…`; the first code stage confirms the list from `bundle.json` (resolvability version 6, a family of `validated_panels.csv`) and records it in the report. With the M3c code: (1) the bundle's spec, re-read from `bundle.json`, gives the same `build_hash` (so no rebuild is triggered and the bundles are reused); (2) `decide()` re-run on the stored `resolvability_cells.parquet` with the stored settings reproduces the stored decision rows of `resolvability.parquet` (canonical CSV sha256 equal, every regime); (3) re-simulating each recipe from the bundle's stored test set gives the same simulated query (sha256 as in (i-a) (1)) as the `83e81e3` code on the same inputs. The HD4 MERSCOPE family has no bundle yet; its version-6 selection is covered by (i-a).
- **Pass:** every hash equal. Any difference fails M3c.

**(ii) Regression: R3 with per-class depth reproduces phase 1's D3 predictions** (pass / fail). Fixtures (small derived tables in `tests/test_annotation/data/`; `m3c_d3_fixtures.json` records each source file, its sha256, the D3 manifest and the derivation):
- `m3c_d3_efficiency_mouse5k.csv`: `gene_id` and log2 efficiency of all 5,006 panel genes, from `$A/5k_real/sim/runs/mouse5k/D3/efficiency.csv` (about 150 KB);
- `m3c_d3_predictions_mouse5k.csv`: from `$A/5k_real/sim/tables/pred_mouse5k_3.csv` (sha256 `de21b164…`), per called class — the 14 major classes (real share ≥ 1%: 01 IT-ET, 02 NP-CT-L6b, 04 DG-IMN, 06 CTX-CGE, 07 CTX-MGE, 09 CNU-LGE, 11 CNU-HYa GABA, 12 HY GABA, 13 CNU-HYa Glut, 18 TH, 30 Astro-Epen, 31 OPC-Oligo, 33 Vascular, 34 Immune) — and the ALL row: `n`, `kish_n`, `cov_class_rule73`, `cov_subclass_rule73` (the provisional-decision columns kept for information);
- `m3c_real_composition_mouse5k.csv`: the public section's called (class, subclass) cell counts from `$A/5k_real/mouse5k/qc/percell_calls_mouse5k.parquet` (the weights of `analyse.py` `comp_weights`).

Two parts:
- **(ii-a) Exact efficiency, CI.** `r3_efficiency` on the 5K panel genes, the 5K mouse `member` asset, table rule `all_measured` and seed 0 equals the fixture's log2 efficiency for every gene. **Pass:** max |Δ| ≤ 1e-9 (the D3 manifest's SD 0.913 ln, 8.7% of genes below 0.25× and 2.0% below 0.125× follow).
- **(ii-b) Replay on the public section** (`scripts/acceptance/m3c_d3_regression.py`, slow). Profile mode (plan §8.3 v7.5) with D3's settings — recipe name `R3_measured_realdepth`, version 1, seed 0, table rule `all_measured`, uniform 25% spill, the public 5K depth-profile asset, M_c = min(max(2 n_c, 2,000), 8 n_c) — on the phase-1 test set (`wmb_selfmap_testset` `6c94b240`, 13,059 cells, not topped up) with the engine of bundle `c30f7eab` and the production mapping configuration on 6 processes, tabulated as `analyse.py` does (class rule: bp ≥ 0.9 and ≥ 20 counts; subclass rule: bp ≥ 0.8, ≥ 50 counts and the class rule; simulated cells weighted to the real called composition as class weight × within-class subclass weight, trimmed at 10× the class median). Profile mode ports the prototype's draw structure, so the replay reproduces D3's simulated cells and calls up to floating-point and chunk-order effects. **Pass:** for each of the 14 major classes and for ALL, at class and at subclass level, |M3c − D3| ≤ 0.01.
- **Why a replay and not a fresh draw.** Two independent draws of D3 differ by sampling alone with an SD of 0.004–0.024 per class (0.0036 / 0.0062 for ALL at class / subclass), so a fresh draw would fail a correct implementation at .01 in most classes; the replay removes draw noise, and (ii-a) checks the efficiency path exactly. Should (ii-b) fail, the report gives each failing cell's sampling floor, which tells a replay defect from draw noise; the pass rule stays |Δ| ≤ 0.01.
- **Reported beside it (no pass / fail):** the same predictions with the production table rule `restricted` against D3 per class (the intended recipe change of plan §8.3 v7.4).

**(iii) Stability of the emitted triples** (pass / fail). On the M3c version-7 5K mouse bundle (panel `4bc22b96…`, WMB, the topped-up test set, the 13-value grid):
- ensemble A = the bundle's emission members {`R1_contam_HO@0`, `@1`, `@2`, `R3_measured_HO@0`}; ensemble B = a fresh keyed draw {`R1_contam_HO@3`, `@4`, `@5`, `R3_measured_HO@1`} (seed 1 for R3's residual, resample and per-cell keys), with the same test cells, grid, mapping configuration and mapping seed 0, run through the diagnostic path (an output directory, never the store);
- triples = the (level, class, bin) emitted in the provisional regime of the unweighted PREP decisions, after the saturated-bp rule and the monotone fill, over every level the bundle decides (broad, class, NT, subclass, supertype), as phase 1 counted them;
- churn = (|A \ B| + |B \ A|) / |A ∪ B|.
- **Pass:** churn ≤ 0.02. Phase-1 reference for two single R1 draws: 47 / 465 = 0.101 (`$A/5k_real/sim/REPORT.txt` §6).
- Reported beside it: the churn per level and for the 14 major classes; the single-member churn R1@0 vs R1@3 under the version-7 conventions (phase 1's analogue); the real 5K provisional coverage (class, subclass) under A and B.

**(iv) Per-class calibration on the public section** (reported; in-sample, never pass / fail).
- Real: the public section's 63,147 vendor cells, with the phase-1 calls (`$A/5k_real/mouse5k/qc/percell_calls_mouse5k.parquet`) when the M3c bundle's query markers equal `c30f7eab`'s (sha256), otherwise re-mapped with the M3c bundle.
- Simulated: profile mode on the public profile for each emission member (R1@0–@2 on their efficiency draws, R3@0 with the `restricted` rule) and the member mean, weighted to the real called composition as in (ii-b); and the class-depth predictor of plan §8.3 v7.5 on the public profile's per-class bin shares (the predictor RESOLVE will use).
- Per major class (the 14 of (ii)) and level (class, subclass): prediction − observed for the raw §7.3 rule and for the coverage under the version-7 ensemble's provisional decisions applied to the real cells (bin of the total counts, applied threshold, floors, mouse subclass floor 60); which (level, class) the per-class warning flags (real < simulated − 0.10); simulated precision (no real counterpart). The ALL row is shown beside the classes, never alone.
- Phase-1 reference (D3): major neurons −.03 to +.09, glia +.06 to +.18. Depth, composition and (for R3) factors come from the predicted section, so every number is in-sample, a best case.

**(v) No family promoted by any M3c change** (pass / fail).
- `validated_panels.csv` and `validated_panel_levels.csv` are byte-identical to `83e81e3`; M3c adds no row and opens no gate-P PR.
- For every family with a bundle in the stores on 2026-09-28 (the version-6 families of (i-b), the 5K mouse `c30f7eab` and 5K human bundles, the public-panel bundles of M3b and the v1 mouse bundle `b0de59dc`), the trust state under the M3c code (on its new version-7 bundle where one is built) is at most its state at `83e81e3` in the order refused < broad_only < provisional < validated, with the validation basis and `validated_max_level` unchanged; both 5K families are `provisional` before and after.
- Property tests: the simulation-input assets and the real-data QC outcomes never raise a trust state (adding, removing or permuting assets, and every QC outcome, leave it equal or lower); a monotone-filled bin never enters the trust tests.
- **Pass:** all three hold.

**(vi) Resources** (pass / fail on memory; wall time reported). The version-7 PREP of the 5K mouse bundle, built into a new build directory of the large store (≤ 8 processes; MapMyCells on 6): peak process-tree PSS ≤ 64 GB, sampled at least every 5 s as M3b measured it. **Pass:** ≤ 64 GB. Reported: the largest single process's RSS, the wall time of every step (reference and query markers, test set with the top-up, each member's simulation and mapping, decisions, tables) with the host load average, and the same for the 5K human version-7 PREP, the fresh-ensemble run of (iii) and the runs of (ii-b), (iv) and (vii). NP9's time reference is OD-E18.

**(vii) Version-7 diagnostic of the version-6 families** (reported; for M8 and the open item of OD-E10; never applied, never written to the store). For set a (WHB 297 and 265, SEA-AD) and for MS1 and MS2: the emitted triples under version 6 (the stored decisions) and under version 7 (R1 × 3), lost and gained per level; the version-7 churn under a fresh draw as in (iii) (R1@0–@2 vs R1@3–@5); and the H18 and H7 proxies under version 7, computed as the M3b follow-ups computed them (`$A/m3b/final_followup/`).

### 21.3 Handling

- Every result goes into the M3c PR and `$A/m3c/M3C_EXIT_REPORT.txt`. A failing pass / fail test needs a fix or your written, recorded approval; a failure is not re-run with other seeds or constants.
- Fixed here and only tightenable: the ensemble members and seeds; the spread limit max(0.03, 3.5 SE) with ≥ 10 confident calls per member; the saturated-bp share 0.90 and cap 0.99; the monotone rule and its 1,000-count non-neuronal limit; the top-up target of 200 test cells; the 13-value grid; the R3 table rules (residual 0.20 log2, cap ±3, the 0.5% top-class share); the fixtures and tolerances of (i)–(vi).
- Not used for tuning: ensemble B of (iii), the public-section calibration of (iv) and the diagnostic of (vii). No version-7 constant may be chosen from them.

## 22. M3c amendment of 2026-09-29 (orchestrator decisions D1–D7, pending your confirmation at PR review)

**Status.** These decisions were made on 2026-09-29 by the orchestrator of the M3c workflow while you were away. They are **orchestrator decisions, pending your confirmation at PR review**: nothing here is your approval, and each can be changed or withdrawn at the M3c PR. The amendment is pre-registered in this documentation commit, on `feature/rca-m3c-5k-measured-efficiency` after the stage-D exit (`46c3bd4`), before any code of the amendment and before any new run (D2). Plan: §8.3 v7.3, v7.7, v7.12, §8.8, §8.10, §12 M3c ("Amendment of 2026-09-29"), decision register D-G15, open decisions OD-E19 and OD-E21.

### 22.1 The stage-D result stays on record

Pre-registered test (iii) **failed** in stage D and the failure stands (`$A/m3c/M3C_EXIT_REPORT.txt` §4 (iii); `$A/m3c/stageD/churn/CHURN.txt`, `churn_anatomy/`): between ensemble A (the bundle `wmb_panel/e0590aac…`: `R1_contam_HO@0`, `@1`, `@2`, `R3_measured_HO@0`) and the fresh ensemble B (`R1_contam_HO@3`, `@4`, `@5`, `R3_measured_HO@1`), churn = (33 lost + 19 gained) / 1,263 = **0.041 > 0.02**; per level broad 3 / 327, class 5 / 258, NT 3 / 243, subclass 13 / 262, supertype 28 / 173; broad to subclass together 24 / 1,090 = 0.022. 40 of the 52 churned triples were emitted in one ensemble only through the member-spread route of E2 with at least one member failing on its own; in the ensemble that emitted them the Wilson bound cleared its limit (target − 0.02) by a median of .017. Nothing of stage D is re-run or re-scored under the amendment, A and B are never used to evaluate it, and the stage-D bundles stay in the store unchanged.

**On tuning (§21.3).** §21.3 forbids choosing a version-7 constant from ensemble B. The *form* of the fix below (more members; a margin on the spread route) was suggested by the stage-D anatomy of A vs B (exit report §4 (iii), options (a) and (b)). Its constants were not fitted: the member count doubles the stage-D ensemble, the margin is one standard error, a conventional round value, and no other value or variant was computed on A or B. Because the form comes from A and B, the amendment is evaluated only on two keyed ensembles never used before (§22.4). This exception to §21.3 is part of what you are asked to confirm.

### 22.2 Decisions (orchestrator, 2026-09-29; pending your confirmation)

- **D1 (churn fix = the exit report's options (a) + (b)).** (a) Eight emission members per version-7 family: `R1_contam_HO` × 6 + `R3_measured_HO` × 2 where a measured factor table exists for the family's species × chemistry, otherwise `R1_contam_HO` × 8; the human 5K lung-ratio recipe stays a gate-P stress recipe only and is never an emission member. (b) A bin can be emitted through the member-spread route only if its pooled Wilson bound clears the limit by at least one standard error (§22.3). Option (c) (decide the report-only supertype level only when fine levels are enabled) is **not** adopted: it changes the test's scope and needs you. Test (iii) keeps its threshold (≤ 0.02) and its scope (every level the bundle decides, supertype included); the broad-to-subclass churn is reported beside it.
- **D2 (pre-registration and evaluation).** This amendment is committed before any new run. It is evaluated only on two new keyed ensembles never used before (not A, not B): the new production bundle's ensemble and one fresh comparator ensemble with a different ensemble key (§22.4). If (iii) still fails, the work **stops** and is reported; no further variant, constant or seed is tried.
- **D3 (stage C open questions).** (1) The monotone fill may re-emit a pool-withdrawn bin whose own measurement passes: the stage-A text (plan §8.3 v7.9) is kept unchanged and the question is listed for you. (2) The version-6 families' diagnostic (§21 (vii)) ran on each bundle's own test set without the top-up: acceptable, recorded.
- **D4 (per-class coverage warning).** The downgrade-only per-class real-vs-simulated coverage warning (plan §8.8) stays on the pre-registered class-depth predictor Σ_d s_c(d) cov(L, c, d) at the dataset's own per-class bin shares; where profile mode has run, its per-class prediction is reported beside it and never decides a warning. The warning is worded per class, not as a "glial" warning: it also fires for the small hypothalamic classes (stage D flagged CNU-HYa GABA, HY GABA, CNU-HYa Glut, HY Glut, MB GABA and MY Glut besides glia and vascular cells).
- **D5 (human lung-scenario headline).** Under a pooled depth scenario without a class composition (the human lung-FFPE scenario, median 245 counts), `annotation-panel-simulate`'s class-depth headline is weighted by the test-set composition (each class's share of the self-map test cells at the level) and labelled "weighted to the test-set composition"; profile mode without a real composition weights its ALL row the same way (its per-class rows are unchanged). A per-class profile keeps its own class shares, and a real composition, when given, keeps precedence.
- **D6 (panel-card glial note).** The note becomes "Simulated glial coverage is an upper bound: -.06 to -.18 on vendor-segmented 5K cells, -.04 to -.14 re-segmented with ProSeg", followed, as before, by "Precision is unmeasured on real data" (evidence: `$A/5k_real/phase1b/REPORT.txt` §5.3; ProSeg 3.2.0 with the vendor XOA cells as its prior, not MerXen's Cellpose-SAM + ProSeg). The ProSeg depth profile is **not** registered as an asset (evidence only). OD-E19's row records that the ProSeg run was done under the M3c stage-T instruction.
- **D7 (5K human bundles).** The 5K human bundles (WHB and SEA-AD) are rebuilt with the final code (new build directories; eight members, the trust guard of v7.7 and the spread margin), which removes stage D's deviation 7.1 (they predate the trust guard).

### 22.3 The amended version-7 ensemble (plan §8.3 v7.3, v7.7)

**Members.** A member's draws depend only on its recipe and seed (R1: the efficiency generator `[seed, 0x6566]` and the per-cell keys; R3: its residual, resample and per-cell keys), so two ensembles with disjoint members are independent keyed draws; an ensemble's *key* is its set of member seeds.

| Family | Emission members (8) | Reported only (never emission) |
|---|---|---|
| Version-7 families whose species × chemistry has a measured factor table (Xenium Prime 5K mouse) | `R1_contam_HO@0`, `@6`, `@7`, `@8`, `@9`, `@10`; `R3_measured_HO@2`, `@3` | `clean@0` |
| Every other version-7 family (Xenium Prime 5K human; custom Xenium v1 or Prime without a table; MERSCOPE, incl. M13's custom panel; unknown) | `R1_contam_HO@0`, `@6`, `@7`, `@8`, `@9`, `@10`, `@11`, `@12` | `clean@0`; human Prime families also the stress recipe `R1_xtissue_lung_stress@0` (reported per bin, as before; gate P's NP6 stress recipe) |
| Version-6 families, diagnostic only (§21 (vii)) | as the row above (no table) | `clean@0` |
| Comparator of the re-test (§22.4; never production) | `R1_contam_HO@20`–`@25`; `R3_measured_HO@20`, `@21` (without a table `R1_contam_HO@20`–`@27`) | — |

`R1_contam_HO@0` stays in production: it is the pre-registered realisation and gate P's NP3 base recipe, which §21.1 keeps a member. Seeds 1–5 of R1 and 0–1 of R3 were drawn for ensembles A and B and are not re-used, so the production ensemble shares only `R1_contam_HO@0` with A, and none of its members with B or with the comparator. "Never a PREP member" (D1) is read as never an emission member of PREP's ensemble: the lung stress recipe stays reported per bin, as in stage D.

**The spread route (E2) with its margin.** For a tested set on which E1 passes and not every emission member emits the bin, let p̂ be the set's point precision and n_eff its effective n at the applied threshold on its check set (the check half; every row in the validated regime) — the p̂ and n that give its Wilson bound L (v7.7 E1: n_eff = Kish n × distinct test cells / rows, the number of distinct confidently called test cells when unweighted). Its standard error is

  SE = √(p̂ (1 − p̂) / n_eff),

the binomial standard error of the pooled precision on the Wilson bound's n (the statistic of NP4 and of E2's spread limit, applied to the pooled set). The bin is emitted through the spread route iff, in addition to the stage-D spread condition (every member ≥ 10 confident check-half calls and max_m p_m − min_m p_m ≤ max(0.03, 3.5 √(p̄ (1 − p̄) / n̄))),

  L − (target − 0.02) ≥ 1 · SE   (float tolerance 1e-9).

With p̂ = 1 the SE is 0 and the margin adds nothing (such a set's Wilson bound depends only on its distinct-cell n). The margin applies wherever E2 judges a tested set: a bin's own pooled set, the ensemble deep pool whose verdict a pooled bin takes, and the pool whose failure withdraws the judged bins at or above D_P (v7.7, as in version 6). A set that passes the spread condition but not the margin fails with the reason `ensemble_spread_margin` (a pool's failure withdraws with `pool_ensemble_spread_margin`). Unchanged: E1, the unanimous route (every emission member emits the bin by its own decision), the spread limit, the saturated-bp rule, the thresholds (t*_pool; the default in the validated regime), floors and trust from the decisions before the fill, the trust guard, and the monotone fill (v7.9): the margin is a condition on the Wilson bound, which the fill waives with the other power conditions, so a bin deeper than the shallowest emitted bin that fails only the margin can still be filled when its point precision and coverage pass at t_d (it is then `monotone_filled`, outside trust and floors, and its cells are `resolvability_extrapolated`; listed for you in §22.6). Each decision row records `spread_se`, `wilson_clearance` (L − (target − 0.02)) and `spread_margin_ok`; the summary counts `n_spread_margin_failed` per regime and records the margin (`spread_wilson_margin_se` = 1) with the ensemble settings.

**Older bundles and the build hash.** A version-7 bundle built before the amendment (its `ensemble_settings` lack `spread_wilson_margin_se`) re-derives its decisions as built (margin 0). The version-7 `build_hash` payload gains `ensemble_rule_version` 2 (stage A–D: version 1), so no bundle decided by the old rule is reused; decision settings stay recorded, not hashed, as before. Version 6 is untouched: its code path, payload and golden hashes do not change (§21 (i)).

**What changes, and how.** The margin only removes emissions (a tightening). More members are not a pure tightening: the pooled estimate and its n are unchanged in expectation but the E2 route changes — unanimity of eight members is rarer, and by sampling alone the range of eight equally precise members exceeds 3.5 SE 20.6% of the time (three 3.6%, four 6.4%, six 13.2%; normal approximation), so the spread route also rejects more bins; gate P scores NP3–NP7 in eight members instead of four or three. Changing the members changes a constant that §21.3 fixes, so it needs your written approval in the M3c PR (§1); until then it is this orchestrator decision, pending your confirmation, with the stage-D members and results on record. The spread limit and its 3.5 SE multiplier are not changed (they were motivated for three or four members; §22.6).

**Cost** (from the stage-D measurements at host load 40–70; [L]). The 5K mouse PREP maps nine members (eight + clean) instead of five: about 7.5 h of self-map (stage D: 14,955 s for five) plus about 1.8 h of build steps, about 9–10 h in all; the decision step's cells table doubles (about 5.2 M rows; stage C measured 4.0 GB at 2.58 M). The comparator ensemble runs about 8 h (stage D's four-member B: 3:58 h). The 5K human PREP about 2.5 h (stage D 80.8 min). Every job keeps to ≤ 8 processes and ≤ 64 GB, at most two heavy jobs at once, CPU only. NP9's time reference stays OD-E18.

### 22.4 Evaluation protocol (D2)

Test **(iii)** is re-tested once, after the amended code is committed and before any other use of the new bundles:
- **P** = the emission members of the new version-7 5K mouse bundle, rebuilt with the amended code into a new build directory of the large store (panel `4bc22b96…`, WMB, the topped-up test set — the test-set bundle `457e0140…` is reused when its build hash is unchanged —, the 13-value grid): `R1_contam_HO@0`, `@6`–`@10`, `R3_measured_HO@2`, `@3`.
- **C** = the comparator `R1_contam_HO@20`–`@25`, `R3_measured_HO@20`, `@21`, simulated on the same test cells with the same grid, mapping configuration and mapping seed 0 through the diagnostic path (`annotation-panel-simulate` / `scripts/acceptance/m3c_v7_diagnostic.py`; an output directory, never the store).
- Triples and churn exactly as in §21 (iii): the (level, class, bin) emitted in the provisional regime of the unweighted PREP decisions, after the saturated-bp rule and the monotone fill, over every level the bundle decides (broad, class, NT, subclass, supertype); churn = (|P \ C| + |C \ P|) / |P ∪ C|. **Pass: churn ≤ 0.02.**
- Reported beside it (no pass / fail): the broad-to-subclass churn (broad, class, NT and subclass together); the churn per level and for the 14 major classes; the single-member churn `R1_contam_HO@0` vs `R1_contam_HO@20`; per ensemble the emitted bins by route (unanimous, spread, filled, saturated) and the spread-margin failures; the real 5K provisional coverage (class, subclass) under P and C.
- **If (iii) fails: stop** and report (D2). No further variant, constant, seed or member count is tried, A and B are not consulted, and the failure goes to you with the numbers.
- Also on the amended code: (i-a) at every commit; (i-b) re-run (no mapping); (v) on the new bundles (every family at most its earlier state; both 5K families `provisional`); (vi) on the rebuilt 5K mouse PREP (peak process-tree PSS ≤ 64 GB; wall times reported). (ii) is unaffected (R3's recipe and table do not change; (ii-a) runs in CI). (iv) and (vii) are reported tests: their stage-D results stay the record; re-running them under the amendment is optional and reported, never a pass / fail.
- **D7:** the 5K human WHB and SEA-AD bundles are rebuilt with the final code (new build directories); their trust states must stay `provisional` ((v)).
- Not used for tuning: C, and every result of the re-test.

### 22.5 Fixed by this amendment and only tightenable

The production and comparator members and seeds of §22.3; the margin of one SE and its definition; the scope and threshold of (iii); the protocol of §22.4. Unchanged and still fixed by §21.3: the spread limit max(0.03, 3.5 SE) with ≥ 10 confident calls per member, the saturated-bp share 0.90 and cap 0.99, the monotone rule and its 1,000-count non-neuronal limit, the top-up target of 200 test cells, the 13-value grid, the R3 table rules and the fixtures and tolerances of (i)–(vi).

### 22.6 Open questions for you (with the M3c PR)

1. Confirm D1–D7 (plan OD-E21), in particular the exception to §21.3's no-tuning rule (§22.1) and the member change (§22.3), which is not a pure tightening.
2. `R1_contam_HO@0` is shared between the production ensemble and stage D's A (it is the pre-registered realisation and gate P's NP3 base): keep, or replace it by a fresh seed (which would need gate P's NP3 base to change)?
3. With eight members the 3.5 SE spread limit rejects equally precise members 20.6% of the time by sampling alone (four: 6.4%): keep the limit, or scale it to the member count (a loosening, so your decision)?
4. The monotone fill can fill a bin that fails only the spread margin (as it fills other power failures) and, as stage C noted, a pool-withdrawn bin whose own measurement passes (D3): keep, or exclude such bins from the fill (a tightening)?
5. "Never a PREP member" for the lung stress recipe is read as "never an emission member": it stays reported per bin in PREP. Remove it from PREP altogether?
6. Option (c) (the report-only supertype level decided only when fine levels are enabled) is not adopted; it would remove supertype, 28 of stage D's 52 churned triples, from the test's scope.

### 22.7 Result of the re-test (recorded 2026-09-30, after the run; changes nothing above)

Test (iii) was re-tested once by §22.4 on the amended code (`6bb42fd`): P = the rebuilt 5K mouse bundle `wmb_panel/f6127077…` (members as §22.3; test set `457e0140…` reused, 14,072 cells; 13-value grid), C = the comparator (`R1_contam_HO@20`–`@25`, `R3_measured_HO@20`, `@21`) on the same test cells, grid, mapping configuration and mapping seed 0 through the diagnostic path. **Result: FAIL**, churn = (24 lost + 38 gained) / 1,235 = **0.050 > 0.02** (|P| 1,197, |C| 1,211). Reported beside it: broad to subclass (13 + 21) / 1,086 = 0.031; per level broad 6 / 329, class 5 / 259, NT 5 / 245, subclass 18 / 253, supertype 28 / 149; single member `R1_contam_HO@0` vs `R1_contam_HO@20` (71 / 1,187) = 0.060. In the ensemble that did not emit a churned triple, the reason was the member-spread limit in 32 of the 62, the new one-SE margin in 13, coverage in 9, the Wilson bound in 4 and no threshold in 4; 53 of the 62 were emitted through the spread route with at least one member failing. By D2 the work **stopped**: no further variant, constant, seed or member count was tried, A and B were not consulted, and the result goes to you (`$A/m3c/M3C_EXIT_REPORT.txt` §10; `$A/m3c/stageE2/churn/CHURN.txt`, `churn_anatomy/prime5k_*`). The other tests on the amended code: (i) pass ((i-a) in CI; (i-b) re-run); (ii) pass ((ii-b) re-run, max |Δ| < 1e-4); (v) pass (every family at most its earlier state; the 5K mouse and the rebuilt 5K human bundles `provisional`); (vi) pass (peak process-tree PSS 28.9 GB for the eight-member 5K mouse PREP, 9.1 h). (iv) and (vii) were re-run and are reported. The failure of both (iii) runs stays on record; the rebuilt bundles stay `provisional` and nothing is promoted.

### 22.8 Decisions of 2026-10-01 (user, after the re-test)

The orchestrator put the question of §22.7 to the user with the options of `$A/m3c/M3C_EXIT_REPORT.txt` §10.7 (keep version 7 as is and report (iii) instead of pass / fail, a loosening; scale the spread limit; option (c); another pre-registered design) and the recommendation to keep version 7 as is. **The user's answer, given in the chat on 2026-10-01:** "accept it as is and report the stability result".

- **Version 7 is accepted as amended (§22.3)**: eight emission members and the spread-route margin, for the families that use version 7. Test (iii) is reported, not scored: churn 0.050 (62 of 1,235 emitted triples) against the registered ≤ 0.02. This is a loosening under §1 rule 2, approved in writing; tests (i), (ii), (v) and (vi) passed and stay pass / fail; (iv) and (vii) stay reported.
- **D1–D7 are confirmed** (OD-E21), with the exception to §21.3's no-tuning rule (§22.1). Accepting the amended version 7 "as is" also settles §22.6's questions 2–6 as they stand: `R1_contam_HO@0` stays shared, the 3.5 SE spread limit is kept at eight members, the monotone fill is kept, the lung stress recipe stays a reported PREP member, and option (c) is not adopted.
- **Test (i) after the merge with M8:** the golden build hashes of the eight human self-map bundles (`seaad_mr_panel`, `whb_frontal_supc_clus` on the four human golden panels) were re-pinned when M3c merged after M8, because M8 D1's held-out test-set revision enters those build hashes by design (§18); the held-out test-set bundles, every mouse value and every version-6 table stay byte-identical. The version-7 self-map now also records D1's `test_set_exclusion`, which its test cells already followed. The four version-7 human bundles built by M3c before the merge (two WHB, two SEA-AD) predate D1; D1's revision enters every human self-map's hashed test-set params, so the merged code builds new ones and never reuses them.
- **Test (i) on other hosts:** the resolvability tables' floats differ in their last bits between the GitHub runner and dwight (same pinned packages), so test (i-a) compares those two tables to 10 significant digits (re-derived on dwight, where the 17-digit values still matched); the decisions, cells, floors, trust and summary stay byte-exact. This relaxes only the comparison of those float columns beyond 10 significant digits; every decision and emitted bin is still compared exactly, so no verdict depends on it.
- **Nothing is promoted** (test (v)): every 5K bundle stays provisional. The current 5K mouse bundle is the amended code's `f6127077…`; stage D's `e0590aac…` stays in the store as history. Nothing reads version 7 until RESOLVE does (the M4 follow-up), which is also where `M3C_EXIT_REPORT.txt` §10.7 question 4 (smoothing emission over neighbouring bins, or reporting the ensemble range) belongs.

### 22.9 RESOLVE reads version 7 (recorded 2026-10-06; the M3c follow-up; no threshold change)

Human and mouse RESOLVE now read resolvability version-7 bundles (plan §12 M3c, "Follow-up for RESOLVE"; chunk C14 of the M13 implementation plan, on `feature/rca-m3c-resolve-v7`). Until now nothing read version 7 (§22.8): the first version-7 family with real datasets, the new-panel human MERSCOPE family (HM1, HM2, HM3, HM4), stopped in RESOLVE on the version guard on 2026-10-02. This is recorded before any version-7 dataset is resolved or scored.

- **What RESOLVE applies** is the version-7 rule as registered in §21 and amended in §22.3, accepted in §22.8: the ensemble decisions re-derived from the bundle's `resolvability_cells.parquet` with each member's cells reweighted to the dataset's composition (per-tested-set trim), the saturated-bp rule, E1 and E2 with the bundle's own ensemble settings (the spread margin included; a bundle built before the amendment re-derives without it, as in PREP), then the monotone fill with its non-neuronal limit. Floors come from the decisions before the fill, the trust state is the bundle's guarded constraint (§21 (v)), and cells of filled bins are `resolvability_extrapolated`. Version-6 families are read exactly as before.
- **Report-only additions** (plan §8.3 v7.5, v7.9): the label column `flag_nonneuronal_high_depth` (a non-neuronal cell at ≥ 1,000 counts in a bin emitted on its own ensemble verdict and marked `nonneuronal_high_depth`; null for version-6 bundles); and, in the resolve summary, the class-depth prediction per (level, called class) at the dataset's own per-class depth. A class with fewer than 100 called cells takes the label-free total-count histogram of all table cells. The value 100 is not pre-registered: the implementation chose it (it reuses the registered per-class minimum of a depth profile, §8.3 v7.5) and it awaits the user's confirmation. This minimum decides no warning while `real_qc.coverage_min_cells` is at least 100 (default 200): the per-class coverage warning (§8.8, user decision 4), which M13 wires, judges only (level, class) pairs with at least that many cells, and those then always take their own depth. Neither addition changes a status, a label, a threshold, a floor or a trust state.
- **Nothing changes:** no threshold, target, margin, floor, grid, member, seed, ensemble constant, trust rule or criterion of §9, §21 or §22. Version-6 RESOLVE outputs are identical apart from the new null column: `tests/test_annotation/test_resolve_v7.py` compares the label tables, provenance and summaries of five version-6 and table-free scenarios (human and mouse) with digests computed with the code at `57c8488`, floats to 10 significant digits as §22.8 allows for test (i-a). A resolvability version the code does not know (later than 7, below 1, not an integer) is refused for every caller.
- **Still open** (for the user, before the new-panel human MERSCOPE family is resolved and scored): `M3C_EXIT_REPORT.txt` §10.7 question 4 (smoothing emission over neighbouring bins, or reporting the ensemble range), which §22.8 placed with this follow-up, is not decided here: RESOLVE applies the ensemble's bins as decided, and the class-depth table it derives keeps each bin's member minimum and maximum. The minimum of 100 cells above. And the non-neuronal high-depth rule of §8.3 v7.9 is keyed on the bin's grid depth: the family's version-7 WHB bundle has a grid that ends at 250 counts, so it has no `nonneuronal_high_depth` bins, `flag_nonneuronal_high_depth` is false for every table cell of HM1, HM2, HM3 and HM4, and glia at ≥ 1,000 counts fall in the open-ended 250 bin, which the fill's non-neuronal limit (bins ≥ 1,000) does not reach; RESOLVE applies the rule as registered. The per-class coverage warning and the rest of the real-data QC wiring belong to M13.

## 23. New-panel human MERSCOPE family: real-data acceptance and gate-P definitions (M13; recorded 2026-10-06, before any measurement on the family)

**Status.** Recorded on 2026-10-06 on `feature/rca-m13-new-panel-onboarding` (cut from the integration branch at `57c8488`), before any measurement on the family and before the set a dry run (§23.9). The source is the M13 planner's draft (`$A/m13/M13_IMPLEMENTATION_PLAN.md` §5, 2026-10-02) as corrected by its CHECK section (K1–K17, which supersede the draft text they correct), with the user's answers of 2026-10-02 (§23.2). The rules of §1 apply to everything below.

**Labels.**
- **[P]**: *proposed; to be confirmed by the user before any of the family's outputs are read.* It marks every threshold and definition that depends on a decision the user has not answered. Where the planner recommended a default, that default is written as the proposal; none is adopted until the user confirms it.
- **[R]**: registered elsewhere (§9, which is plan §14; plan §8.8; §18–§22) and only cited here. Nothing in this section changes it.
- **[U]**: decided by the user (§23.2).

**Reading rule.** No output of the family is read, the label-free depth read included, until the user has confirmed, changed or withdrawn every [P] item of §23.3–§23.7. The [P] items of §23.9 are also answered before the set a dry run runs. Each answer is recorded here, dated, before the read or run it governs.

### 23.1 The family, and what exists before this section

- **The family** (the new-panel human MERSCOPE family): four human brain sections, HM1, HM2, HM3 and HM4, from four donors (D26), each imaged on MERSCOPE only (unpaired). One Vizgen custom panel ("HumanBrainV2", codebook C2P2452): 496 genes and 65 `Blank-*` codewords, with an identical `cell_by_gene` header in all four sections.
  - Provisional `panel_hash` `dcddfbd18fb8…` and `family_id` `human_merscope_dcddfbd18fb8` (basis `own`); Jaccard with set a 0.108 (77 shared genes).
  - The family is in neither `validated_panels.csv` nor `resolvability_v6_pins.csv`, so it resolves to resolvability version 7 (§21.1) with the eight members of §22.3: `R1_contam_HO@0`, `@6`–`@12`, with `clean@0` reported. MERSCOPE has no measured factor table, so there is no R3 member; it is not a Prime family, so there is no lung stress recipe.
- **What exists on 2026-10-06, unread:**
  - the user's legacy-mode pipeline outputs for the four sections (MerXen segmentation, spatial data and the legacy annotation). The legacy labels exist and have not been read;
  - the outputs of a `map_first` run of 2026-10-02: integration-branch code at `57c8488`, `anatomical_region = frontal_cortex` set explicitly in a samplesheet copy, all four segmentations, cortical depth off, written add-only with the table suffix `m13` under the M8 safeguards (store backups, before / after manifests). Its PREP built the family's version-7 bundle and its MAP tasks completed. All 16 RESOLVE tasks (4 sections × 4 segmentations) failed with `ResolvabilityError` ("a resolvability version-7 bundle … needs a consumer that declares version-7 support"): RESOLVE has no version-7 consumer yet (M13 plan chunk C14; plan §8.3 v7; §21–§22). The family's `map_first` labels therefore do not exist. Its MAP outputs exist under the family's results and have not been read. The run's own instructions forbid reading the family's outputs until this section is committed; only its logs (task states and the error message) have been read.
- **Before any measurement.** Nothing here was chosen with an annotation or MerXen-segmentation measurement of the family in view. What was computed before this section, all label-free, is disclosed in §23.8.

### 23.2 Decisions of 2026-10-02 recorded with this section

The user's answers, given in the chat on 2026-10-02 and recorded by the M13 orchestrator, verbatim where quoted (`$A/m13/M13_IMPLEMENTATION_PLAN.md`, "USER DECISIONS"). Every other decision the planner put (D5–D24, D27) and the CHECK's D28 and D29 are **unanswered**. Their recommended defaults are not adopted, and every item below that depends on one is [P].

| # | Question (planner's §4) | Answer | Label (§1 rules 1–4) |
|---|---|---|---|
| D1 | Does the M8 other-region COP drop (§18 D1) apply to M13, gate P and the human families? | "use for all new panels": option (a) | loosening (rule 2), approved in writing |
| D2 | Other-region test cells in the leave-one-donor-out sets | "Yes, do this": option (d), a true per-donor hold-out, with (c) as the fallback | definition of the replicates; no threshold |
| D3 | The family's anatomical region | "Yes it is frontal cortex.": option (a), all four sections | input; no threshold |
| D4 | The dry run before a new family is scored | "Dry run should only be against panels from the same species as the new panel. The species must be selected when starting a new panel check." | the planner's D4 called this an ordering amendment; CHECK K2 classes it as a loosening (rule 2: it drops MS1 and MS2 from the dry run before a human family is scored). The user's rule is in writing; that it approves the loosening is to be confirmed by the user in this family's gate-P PR |
| D25 | Compute for the family | "Yes lets go ahead, please prepare for this and I will tell you when to start" | compute approval; the user starts each run |
| D26 | The family's metadata | quoted below | inputs; no threshold |

- **D1 (a).**
  - The M8 D1 rule (§18: leave out the other-region test cells whose truth supercluster is Committed oligodendrocyte precursor) applies to every human gate-P test set and to the PREP self-map of every new human family, this one included. The gate-P test sets are the default donor's check half and both other donors' leave-one-donor-out sets (D2), with their stress and clean replicates.
  - It is what the code does today (`reference.self_map_test_cells` applies it to every self-map on `whb_frontal_supc_clus_ho`), so it needs no code change and no rebuild. It settles §18's "Not covered" item for gate P, 5K and the M13 custom panels. Options (b) (key the drop to the M8 bundles) and (c) (also a no-drop set a dry run) were not chosen.
  - §18's caveats C1 and C2 carry over, and the gate-P PR restates them with the approval: the set a evidence behind D1 (PREP broad Oligodendrocyte precision) does not carry over to other panels; on set a, the other-region and donor COP cells are called Oligodendrocyte at rates that do not differ per depth (Fisher p = .36–1.00); with the drop, the COP → Oligodendrocyte error is estimated from the donors' frontal COP cells only (20 in the default donor on set a), so supercluster COP may be `not_evaluable` in gate P.
- **D2 (d).**
  - Each of the two other frontal WHB donors (H19.30.001, H18.30.002) draws its own non-frontal non-neuronal top-up cells, excluding every cell of the default held-out test set, under the same per-class top-up rule. The default donor's (H19.30.002) replicate stays exactly the production check half, and the frozen thresholds still come from it. The replicates are then disjoint: no cell is counted twice, and no fit-half cell leaks.
  - The per-donor pool sizes are reported from the reference metadata (no mapping) before the first leave-one-donor-out build. If a donor's own pool is too small, the user decides the fallback (c) with those sizes in view, before any replicate is mapped. Under (c) the shared cells stay in every replicate, each cell is counted once in pooled sets, and the default donor's fit-half cells are excluded everywhere. What counts as "too small" is [P].
- **D3 (a).** All four sections are frontal cortex, and every run sets `anatomical_region = frontal_cortex` explicitly (the run of 2026-10-02 did). NR11 is therefore scored (§23.6). Gate P's NP7 region part uses `region_plausible_frontal_cortex` whatever the sections' region, because gate P simulates from the frontal reference (CHECK K4).
- **D4.**
  - The dry run uses only the seeded families of the new family's species: set a before a human family is scored; MS1 and MS2 (with the second disjoint WMB draw) before a mouse family. The gate-P command takes a required species argument, chosen when a new-panel check starts.
  - This amends §9's "Dry run" ("Before any new family is scored, the programme runs on the seeded families (set a, [MS1], [MS2])") per species; §9 stays the verbatim copy. The planner's D4 called this an ordering amendment with no threshold change; CHECK K2 classes dropping a dataset from the dry run as a loosening under rule 2. That classification is the check's, not the user's: the user's rule quoted above is in writing, and this family's gate-P PR quotes it and asks the user to confirm it as the written approval of that loosening.
  - Unanswered: whether the human Prime 5K and Xenium Human Brain (266) gate-P runs join this pass (the planner's D4).
- **D25.**
  - Approved: the family's PREP, the `map_first` annotation of the four sections, and gate P on the family. The user starts each run.
  - Run settings the user asked for on 2026-10-02: `map_first` written add-only with a suffix into the family's existing results, with the legacy outputs untouched and the M8 safeguards; all four segmentations; the M8 steps (map-first labels, MENDER, cortical depth, annotation report). Cortical depth and its report items are added once the user's boundaries exist.
- **D26.**
  - Answers: "They are four different donors." "Diagnosis unknown, assume the same as the other batch of four for now" (HD1, HD2, HD3, HD4). "The tissue is fresh-frozen, rather than the FFPE of [HD1], [HD4], [HD2], [HD3]." "The antibody targets are named in the channel names." On the pia / WM boundaries: "Yes, I will draw that for all four once they are processed, before you start running anything", so the H12-style depth check is in scope for all four sections (NR15).
  - Effects: NR14's spread across sections is a spread across donors. The gate-H analogue thresholds proposed in NR10–NR12 were measured on the FFPE sections of the other batch, and this family is fresh-frozen.
  - Not answered: Braak / CERAD, age, sex and PMI; section thickness; the expected grey / white matter share and rare classes; whether GFAP, SNAP25, MBP and CLU are protein IF; whether the panel will ever be paired with another panel (NP8).

### 23.3 Rules

1. Real data never promote this family (OD-E1 / OD-E9) [R]. Every check here can only warn, lower a section's gate level, force `broad_only` or `not_resolvable`, or open a review. A passing check changes neither the trust state nor the margins.
2. The family is not seeded, so §8.8's downgrades apply in full, with no warn-only period [R]. The only exceptions are the ones this section writes (NR5's effect, D18; NR9's effect, D23), each [P].
3. Nothing below is revised after any of its values is seen, except by a tightening or by a loosening the user approves in writing and the PR quotes (§1 rule 2).
4. Promotion is possible only through the family's own gate-P PR (§9, gate-P rule), which quotes this section's outcomes. The family's real-data QC runs and is reviewed before any gate-P PR for the family merges; otherwise precondition P2 would fail (CHECK K16).
5. Repository texts name the family only by its neutral name and the section ids.

### 23.4 Datasets and configuration

- **Datasets:** HM1, HM2, HM3 and HM4 (human, MERSCOPE only, unpaired, four donors).
- **Segmentation:** proseg_hybrid is scored, as the human default (OD-B7′ = A, §20; D25). reseg is reported; proseg_mask and original_seg run and are reported as context.
- **Region:** `frontal_cortex` (D3 (a)) [U].
- **References:** WHB frontal as the primary, SEA-AD as the second vote (the human defaults) [R].
- **Panel** [P]: the declared panel, frozen under D17 (unanswered: the vendor codebook file, an RGS5 override, or symbol resolution without RGS5).
  - The frozen `panel_hash` is recorded here, dated, before any output is read, and confirmed on a published clustered H5AD from its `var` only.
  - The run of 2026-10-02 used the panel as the pipeline derives it, with RGS5 unresolved. A choice that changes the resolved gene set changes the hash, and PREP and MAP are then re-run.
- **Mode:** `map_first` with the version-7 bundle (R1 × 8, §22.3) and `provisional` trust [R].
- **Run settings** [P]: the seeds and `n_processors` are fixed and recorded in the PR; every self-map maps with 8 MapMyCells workers, as in §18 item 2.
- **Code:** the integration-branch code at a recorded commit that includes the RESOLVE version-7 consumer (C14), run with a separate work-dir; outputs add-only with a suffix (D25).

### 23.5 Preconditions [P]

These are mechanical. If any fails, every row is `UNSCORED`.

- **P1.** All four sections carry the same frozen `panel_hash`. All 65 `Blank-*` codewords are removed by the registry, and none reaches the MapMyCells query (NP1, §8.4) [R].
- **P2.** The family has no `validated_panels.csv` row, and its trust is `provisional`.
- **P3.** RESOLVE accepts the version-7 bundle (C14), and each MAP and RESOLVE run names its bundle's build hash. C14 keeps the version-6 and M8 RESOLVE outputs identical apart from its new null schema column, under a regression test (CHECK K11).
- **P4.** The real-data QC module records an outcome for every §8.8 check, per section × segmentation: `pass`, `warn`, `fail`, `not_applicable` or `not_evaluable`.

### 23.6 Criteria

| ID | Criterion | Threshold | Effect when it fails | Kind | Status |
|---|---|---|---|---|---|
| NR1 | Downgrade-only mechanics | Compared with a QC-free re-run of RESOLVE on the same MAP output: the trust state, `EmissionPlan`, `FloorPlan` and every emitted label are identical, except labels set to `not_resolvable` and the gate level lowered by a named check. Paired concordance, the factor re-measure and the prefilter spot check are recorded `not_applicable` (no paired section, no R3 member, 496 ≤ 1,000 genes; §8.8) | The M13 exit fails for the family (a code defect) | must pass | definition [P]; the principle [R] (§8.8) |
| NR2 | Reproducibility (H15 analogue) | (a) An identical re-run gives identical label-table content. (b) MapMyCells `rng_seed` 0 vs 1: confident broad labels switched between two confident names ≤ 1% of the seed-0 confident broad labels (gate H's reading, §20 D12 (a)); the threshold-crossing share is reported. This reading is more lenient than the planner's draft ("seed 0 vs 1 changes ≤ 1% of confident broad labels"), which, read literally, also counts the labels that cross the confidence threshold; gate H measured 7.0% and 13.4% crossings on HD1 and HD2 (§20 D12 (b)) | (a) The M13 exit fails. (b) A warning, listed in the PR | (a) must pass; (b) warning | [P] |
| NR3 | Resources (H14 analogue) | Annotation wall ≤ 2.5 h per section over the evaluated segmentations; MAP peak RSS ≤ 16 GB; no GPU; PREP wall time against NP9's reference (D10 / OD-E18, open) | A warning, listed in the PR | warning | [P] |
| NR4 | Dataset gate (§5.4, gate A) | As defined: `broad_only` when A < 0.30; `failed` when confident broad coverage of table cells < 0.25; a warning when coverage of segmented objects < 0.15 | As defined; a `failed` section is a family-level concern (§23.7) | QC as defined | [R]; the concern trigger [P] |
| NR5 | Marker referee on confident `ct_broad` (§8.8) | A warning below 0.75 and `broad_only` below 0.70 (§8.8's human values, which came from the H9 hand lists). Proposed statistic: G2-style pseudo-label consistency on per-panel marker sets derived from WHB profiles (D18 (a)). `not_evaluable` with fewer than 2 marker groups of ≥ 3 markers, or too few marker-pseudo-confident cells. The derived statistic is re-measured on set a before the family is scored; if set a falls below 0.75, the thresholds go back to the user (D18) | A warning; below 0.70 a gate-level cap (D18 (a)) or a trust cap (D18 (b)) | QC | the values [R]; the statistic, marker sets, effect and `not_evaluable` rule [P] (D18) |
| NR6 | Flag rates (H16, §8.8) | A warning when more than half (`uninformative_strata_warn_frac` 0.5) of a section's (class × platform) strata are uninformative under H16's 15% marking: the literal reading of §8.8 (CHECK K6). D22's option (a) (each flag's own null switch over flag × class × platform) is reported beside it and adopted only with written approval | A warning | QC | the threshold [R]; the reading [P] (D22) |
| NR7 | Gene complexity (§8.8) | A warning, with the "simulated coverage unreliable" text, when native median `n_genes` > 1.45 × simulated in any depth bin with ≥ 50 cells of each kind. Proposed source: simulated `n_genes` stored by the version-7 PREP and counted on the bundle's query genes, native `n_genes` counted on the same genes (D19 (a)). `not_evaluable` when the bundle stores none; the bundle of 2026-10-02 predates the artefact | A warning | QC | the threshold [R]; the simulated cells' source and the `not_evaluable` rule [P] (D19) |
| NR8 | Per-class coverage against simulation (version 7; §8.8, §21.1 item 4, §22.2 D4) | Per (level, called class) with ≥ 200 section cells: a warning when real < simulated − 0.10, on the class-depth predictor at the section's own per-class bin shares. Non-neuronal high depth (`flag_nonneuronal_high_depth` and the trend) is report-only | A warning | QC | [R]; needs C14 for the section's per-class depth |
| NR9 | Registration G1 (M0a, §7.6) | As defined there | §8.8's effect (`failed` + `exclude_hard`), or warn-only in M13 (D23 (b), a loosening against §8.8) | QC | the effect [P] (D23) |
| NR10 | COP control (H5 analogue) | Confident COP supercluster ≤ 2% of table cells; confident broad OPC ≤ 10%; the COP-derived OPC share reported | Listed; a family-level concern (§23.7) | acceptance | [P] |
| NR11 | Region plausibility (H2 analogue) | `flag_implausible` ≤ 1% of table cells; scored, because the region is frontal cortex (D3 (a)) | A family-level concern | acceptance | the threshold [P] |
| NR12 | WHB–SEA-AD 7-class agreement on table cells with ≥ 20 counts (H3 analogue) | ≥ 0.80 (H3's bar for HD2, HD3 and HD4) | A family-level concern | acceptance | [P] |
| NR13 | Confident broad coverage and gate level per section | Report-only: no per-dataset E2 baseline exists for this panel. The predicted levels are written into the PR from the label-free depth read, before any MAP or RESOLVE output of the family is read. Proposed depth source: the MerXen tables of the evaluated segmentation, `total_counts` only, pooled and per section (D8 (b)) | none | report | [P] (D8) |
| NR14 | Composition per section and its spread across the four sections (four donors, D26); reseg vs proseg_hybrid agreement on shared cell ids (M8b tooling) | Report-only | none | report | [P] |
| NR15 | Optional extras, each only if its input exists: held-out-gene enrichment with a panel-specific marker list derived by the §8.6 specificity rule and fixed before the run (D27); protein IF vs RNA concordance per channel, with the targets taken from the channel names (D26); the H12-style depth check on all four sections once the user's pia / WM boundaries exist (D26): Upper-layer IT < Deep-layer IT < Deep-layer NP/CT/6b medians where the gate emits supercluster, and Oligodendrocytes WM > GM, with gate H's scored CIs (tangential blocks for the ordering, §20 D11; square tiles for WM − GM) | Report-only in this first family | none | report | [P] |

### 23.7 What counts as a pass [P]

- **Onboarding accepted** (the family's M13 exit on real data): P1–P4 hold, and NR1 and NR2 (a) pass, on all four sections × proseg_hybrid. Every other row is recorded with the value, threshold and effect it names, and the PR explains each warning.
- **Family-level concern.** It holds back the family's gate-P PR until the user has reviewed it, and it never promotes or demotes on its own. Any of these triggers it:
  - NR4 `failed` on any section;
  - NR5 `broad_only` on ≥ 2 of the 4 sections;
  - NR10, NR11 or NR12 failing on ≥ 2 of the 4 sections.
- **A gate-P pass together with a family-level concern:** the PR states both, and the user decides.

### 23.8 Disclosed prior knowledge (before this section was recorded)

Computed on 2026-10-02 from vendor files and directory listings only. All of it is label-free; none of it is an annotation or MerXen-segmentation measurement.

- **The panel:** 496 genes plus 65 blanks; 77 genes shared with set a; 495 of 496 resolved (RGS5 is unresolved because it is ambiguous in the WHB gene table); no codebook file on disk.
- **Vendor `cell_by_gene` summaries:**
  - HM1: 178,602 cells, median 309 counts and 107 genes;
  - HM2: 119,617 cells, median 144 counts and 59 genes;
  - HM3: 154,962 cells, median 333 counts and 119 genes;
  - HM4: 132,317 cells, median 280 counts and 91 genes;
  - blank share of counts 0.62–0.77%; 31–62% of vendor cells above 250 counts; vendor cells hold 46% of the transcripts on HM4.
- **Region evidence put to the user for D3:** pseudobulk detection of layer markers (CUX2, RORB, CBLN2, TSHZ2) and very high PLP1: neocortex with white matter, not cerebellum.
- **NP2 preview** from the legacy whole-WHB marker lookup (reference data only): 131 root markers; every WHB root child has ≥ 10 markers except Bergmann glia (0) and Upper rhombic lip (9). NP2 is recomputed on the production frontal bundle's lookup. Whether weak or collapsed parents are accepted is the user's call in the gate-P PR (CHECK K14).
- **Run state:** which pipeline steps had published outputs per section (directory names only), and the logs of the run of 2026-10-02 (task states and the RESOLVE error).

### 23.9 Gate-P definitions recorded with this section (every family; before the set a dry run)

Gate P is simulation only (§9). These definitions apply to every family, this one first. Each is [P] unless marked otherwise. The NP4 code (M13 plan chunk 1) implements item 3 and marks its D12 part as pending.

1. **C_P.** Per level, C_P is fixed on the truth parent class of the pooled test cells, before any cell is mapped (D12) [P]. The ≥ 700 test cells and ≥ 90% share rules are [R].
2. **Class-balanced weights** give equal total weight to each truth type within the called class's tested set (D12) [P].
3. **NP4.**
   - *Tested sets:* the pooled seed-0 sets (`gate_p_tested_sets`) applied to each replicate. A "≥ D_P" set takes each test cell once, at its deepest row of the level, before any class or confidence filter (D12) [P].
   - *Floor:* each replicate with ≥ 100 confident calls in the set has point precision ≥ target_L [R].
   - *Vacuous sets:* a tested set where some replicate has fewer than 100 calls is scored by the floor alone, and one where no replicate has 100 passes NP4 vacuously; the NP4 table reports it (`vacuous`) [R] (§9's literal reading). For the user (with D12 and D29): a pooled set that just reaches 200 calls over three donors holds about 67 per donor, so NP4 is often vacuous on pooled sets, and such a set counts as an NP4 pass in the every-member verdict.
   - *Range:* applied only when every replicate has ≥ 100 calls [R]. Per group (a donor or a draw), p_g and n_g are the arithmetic means over its seeds of the per-seed precision and confident n; p̄ and n̄ are the means of the group values; the limit is max(0.03, 3.5 × √(p̄(1 − p̄)/n̄)). The formula is [R]; the averaging is [P] (D12; CHECK K9 item 3). It follows the registered E2 convention (`member_spread`, §22.3).
   - *Weights:* NP4 precision is unweighted, because §14 reweights NP3 only (CHECK K9 item 2) [P].
   - *Seeds:* "seeds 0 / 1" are MapMyCells mapping seeds, scored as call changes (D6, folded into D12) [P]. Seed 0 vs 1 changes ≤ 2% of confident labels per validated level [R].
   - *Every tested set:* an NP4 failure at any tested set fails the (level, class). This is the literal reading of §9 NP4 (CHECK K10). Letting a shallow NP4 failure only raise `validated_min_depth` (D29) would be a loosening.
   - *Inputs:* each replicate is scored from a table that holds exactly one replicate, and the replicates form a full grid (≥ 2 groups, the same seeds in every group). The default donor is scored on its check half only, at every seed, in the tested sets and in NP4, because the frozen thresholds were fitted on its fit half [R] (§9 "Pooled held-out calls"); a fit-half test cell of it in another replicate is a leak (D2 (d)). §9 says "mouse, both draws"; scoring the draw the thresholds came from on its check half in the same way is [P]. Otherwise gate P stops with an error.
4. **NP6.**
   - *Drop test:* fail when the one-sided 95% lower bound of p_base − p_stress exceeds 0.05, from a two-proportion z-test on each set's Kish n (D12) [P].
   - *Factors:* the cross-platform factor source and the rule for genes the table does not cover are D7 [P]. Only 77 of this family's 495 genes have measured offsets. The proposal is the registered basis (`research/xplat` `log2_xen_over_mer_gene_means.csv`, D-A7) with a keyed empirical resample for the uncovered genes, centred on the median gene, capped at ±2 log2 and registered as a `sim_inputs` stress asset with provenance.
   - This family has no R3 member and no lung stress recipe.
5. **NP7.**
   - *Single wrong node:* the share of truth class c's confident calls at a tested set that land on one wrong node; the called-class view is reported (D12) [P].
   - *The 1% denominator:* per level, the pooled held-out confident calls at the frozen thresholds over all classes and emitted bins, per member for version 7 (CHECK K9 item 1) [P].
   - Calls to sink or region-implausible nodes (rows whose `parent` is null, which lie outside the tested sets) are counted separately against that denominator. The region column is `region_plausible_frontal_cortex` (CHECK K4).
6. **NP5 expected depth** (D8) [P]: either each class's median in a per-class depth profile registered as a `sim_inputs` `profile` asset (provenance: the source sha256 and the deriving script; `trust_effect: none`; frozen and recorded before gate P runs on the family; CHECK K7), or the label-free pooled median of the family's sections. No other real-label input reaches gate P.
7. **D1 and D2:** D1 (a) and D2 (d), §23.2 [U].
8. **Dry run** (D4, §23.2) [U].
   - For a human family the dry run is set a only. It must pass at broad for every C_P class and at supercluster for the classes H18 expects ([R]: §9 "Dry run"; CHECK K1).
   - *The classes H18 expects* (D28) [P]: H18's set (every class with ≥ 50 test cells), less only what the user has already approved: the bins and classes that M8 D4 names, and supercluster COP, which D1 removed from H18's scope (§18 C1). Any further narrowing is a loosening.
   - *Regime* (D5) [P]: the proposal is both version 6 (set a's R1@0) and the version-7 ensemble that this family uses, each at frozen provisional-regime thresholds.
   - With D1 (a), every dry-run set carries the drop.
   - A failure means the criteria are unattainable. They are then revised in a PR the user approves, before this family is scored (§9).
9. **NP9** [P]: the identity re-run covers PREP's bundle and one seed-0 replicate per member, with the tables compared (CHECK K13). The time reference for version-7 families is D10 / OD-E18 (open). The prefilter part does not apply at ≤ 1,000 genes [R].
10. **NP8** does not apply unless the family will be paired with another panel (D26, unanswered on this point).
11. **Records** (D14) [P]: no `validated_panel_levels.csv` row is written for classes where nt does not apply, so the rank rule of `validated_max_level` stays unchanged; `ct_seaad_subclass_validated` is always false and report-only, because gate P scores the primary reference only.

### 23.10 Decisions of 2026-10-06 (user; recorded before any output of the family is read)

**The user's answers**, given in the chat on 2026-10-06 and recorded by the M13 orchestrator: "D1: a"; "D2: go with the recommendation"; "D3: go with the recommendation, it is now known that this current data is frontal cortex"; "the rest please go with the recommendation". "The rest" covers every other decision the planner put (D4–D27) and the CHECK's D28 and D29. What it adopts is the planner's recommendation as corrected by the CHECK (K1–K17), because the CHECK supersedes the text it corrects. The full table is in `$A/m13/DECISIONS_RESOLVED.md`.

**Effect on the labels.** Every [P] item of §23.2–§23.9 is confirmed as written, except the items under "Still open" below. Those keep their [P], and their defaults are stated. Confirmed items count as [U] from this date. No threshold, target, margin or registered criterion is changed by this subsection.

**Confirmed.**
- **§23.2.**
  - D1 (a), D2 (d) and D3 (a) are confirmed again in the user's words above.
  - **D4:** the user's rule of 2026-10-02 (§23.2) stands. The dry run uses the new family's species only, and the gate-P command takes a required species argument. Together with "the rest please go with the recommendation" (planner's D4 (a)), it is the written approval of the loosening that CHECK K2 identified, and the family's gate-P PR quotes both answers. The human Prime 5K and Xenium Human Brain (266) gate-P runs join this pass after this family; the Prime 5K Human run needs the lung stress recipe in NP6 (CHECK K8). For mouse families, M13's registered exit still needs the MS1 and MS2 dry run and regression rows.
- **§23.3 rule 2:** the two exceptions are NR5's effect (D18 (a)) and NR9's effect (D23 (b)).
- **§23.4:**
  - the run settings (seeds and `n_processors` fixed and recorded; 8 MapMyCells workers for every self-map);
  - D25's segmentation choice: proseg_hybrid scored, reseg reported. Gate P runs on the family only after the set a dry run passes.
- **§23.5:** P1–P4 as written. P1 waits on the frozen panel (still open).
- **§23.6:** NR1–NR15 as written, with these readings:
  - **NR2 (b):** gate H's reading (§20 D12 (a)). The planner's draft was never registered, so this is not a loosening of a registered criterion.
  - **NR3:** PREP is judged against NP9's time reference (D10 (a), below).
  - **NR5:** D18 (a) and (a): derived per-panel marker sets and a gate-level cap below 0.70, with trust and margins unchanged. The derived statistic is re-measured on set a before the family is scored. If set a falls below 0.75, the thresholds go back to the user before any number of the family is seen.
  - **NR6:** the literal reading (CHECK K6). Option (a) of D22 is reported only.
  - **NR7:** D19 (a). The family's bundle of 2026-10-02 predates the stored simulated `n_genes`, so NR7 is `not_evaluable` unless the family's PREP is rebuilt with that code (compute the user starts).
  - **NR9:** D23 (b), warn-only in M13. It moves to `failed` + `exclude_hard` (§8.8) once the set a regression shows no false G1 failure. The move is a tightening, decided on that regression and recorded here before any real-data QC output of the family is read.
  - **NR10–NR12:** the thresholds as written.
  - **NR13:** depth from the MerXen tables of the evaluated segmentation, `total_counts` only, pooled and per section (D8 (b)).
  - **NR15:** marker sets derived by the §8.6 rule and fixed before the run (D27 (a)), with the hand-curated alternates (b) reported.
- **§23.7:** the pass rule and the family-level concern triggers as written.
- **§23.9**, the gate-P definitions for every family. These are the definitions of the M13 plan's §23.7 draft, with the CHECK's additions:
  1. C_P per level is fixed on the truth parent class before mapping.
  2. Class-balanced weights give equal total weight to each truth type within the called class's tested set.
  3. **NP4:**
     - pooled seed-0 tested sets, each cell once at its deepest row;
     - p̄ and n̄ are the means of the seed-averaged group values, each the arithmetic mean over seeds;
     - unweighted precision;
     - mapping seeds scored as call changes (D6);
     - the default donor scored on its check half, and the draw the thresholds came from scored the same way;
     - an NP4 failure at any tested set fails the (level, class) (D29: the literal reading; the alternative of only raising `validated_min_depth` is not adopted).
  4. **NP6:**
     - the drop test: fail when the one-sided 95% lower bound of p_base − p_stress, from a two-proportion z-test on each set's Kish n, exceeds 0.05;
     - the factors (D7 (a) + (i)): `research/xplat` `log2_xen_over_mer_gene_means.csv`, with a keyed empirical resample for the genes it does not cover, centred on the median gene, capped at ±2 log2 and registered as a `sim_inputs` stress asset with provenance. The X1 factor table is reported only.
  5. **NP7:**
     - the single wrong node is the share of truth class c's confident calls on one wrong node;
     - the 1% denominator is the pooled held-out confident calls per level over all classes and emitted bins, per member for version 7;
     - the region column is `region_plausible_frontal_cortex`.
  6. **NP5 expected depth** (D8 (b) with CHECK K7): each class's median from the family's first provisional `map_first` run's confident broad calls, with the overall median for classes without calls. It is written as a `sim_inputs` `profile` asset (source sha256, deriving script, `trust_effect: none`) and frozen and recorded before gate P runs on the family. No other real-label input reaches gate P.
  7. D1 (a) and D2 (d).
  8. **The dry run** (D4, D5 (c), D28):
     - set a only, on version 6 (R1@0) and on the version-7 ensemble, each at frozen provisional-regime thresholds;
     - pass at broad for every C_P class and at supercluster for the classes H18 expects, defined as in item 8 of §23.9.
  9. **NP9:**
     - the identity re-run as in §23.9;
     - the time reference for version-7 families is 1.5 × the time measured in the set a version-7 dry run, scaled per simulated cell, measured before any number of the family (D10 (a); OD-E18).
  10. NP8 is `not_applicable` (see "Still open").
  11. D14 (a) and (a).
- **Also adopted** (plan §4):
  - D9 (a): the registered grid; the per-class share of cells above 250 counts is reported;
  - D11 (b): the gate-P bundles go in a separate store under the M13 evidence directory, and the production store is written only by the user-started `map_first` runs;
  - D13 (a): the C7 writer also refuses a family without a self-map;
  - D15 (a);
  - D16: the hash-derived `family_id` of the frozen panel and `panel_id` `human_merscope_hbv2_496`;
  - D20 (b), with a per-species record;
  - D21 (shared_mask, point estimate);
  - D24: done.

**Loosenings and their written approvals** (§1 rule 2; the family's gate-P PR quotes them):
- **D1 (a):** "use for all new panels" (2026-10-02); "D1: a" (2026-10-06).
- **D4** (CHECK K2): "Dry run should only be against panels from the same species as the new panel. The species must be selected when starting a new panel check." (2026-10-02); "the rest please go with the recommendation" (2026-10-06).
- **D10 (a)** (NP9's time reference re-baselined) and **D23 (b)** (G1 warn-only against §8.8's table): approved only by "the rest please go with the recommendation" (2026-10-06).

**Correction of fact (§23.1, §23.4) and disclosure (§23.8).** On 2026-10-06 the M13 Decide step read the panel-stage records of the run of 2026-10-02. These hold gene symbols, Ensembl IDs, their resolution sources and the hashes, all derived from the prepared `var`; they hold no count, label or `obs` column. They were read before this subsection was recorded, as the hash check of P1.
- All 496 genes were declared: 491 native IDs and 5 from the fallback table, with none unresolved.
- RGS5 resolved to ENSG00000143248, the first of the WHB gene table's two RGS5 entries (the other is ENSG00000232995). §23.4's "with RGS5 unresolved" is therefore wrong.
- All 16 section × segmentation records carry `panel_hash` `aa25d5a241d0b273…` and `family_id` `human_merscope_aa25d5a241d0`.
- The provisional `dcddfbd18fb8…` of §23.1 came from the planner's own symbol resolution, which left RGS5 unresolved (495 genes). It is not the hash of the run.

**Still open** ([P] kept; the code's behaviour until each is answered):
1. **D17, the declared panel.** There is no codebook file on disk, and no RGS5 choice has been made. The default is the panel as the pipeline derives it (496 genes, RGS5 = ENSG00000143248, `aa25d5a241d0…`): option (b) with the canonical ID, without an override row. Until the user confirms it, supplies the codebook or chooses otherwise:
   - no output of the family is read, the depth read included (reading rule);
   - gate P does not run on the family.

   The gate-P code and the set a dry run do not depend on it.
2. **D2, what counts as "too small".** The per-donor pool sizes are reported from the reference metadata before any leave-one-donor-out build. If a donor's own pool cannot meet the per-class top-up rule, gate P stops before any replicate is mapped, and the user chooses (d) as it stands or the fallback (c) with the sizes in view.
3. **D26, the remaining metadata:** Braak / CERAD (assumed as HD1, HD2, HD3 and HD4 for now), age, sex and PMI, section thickness, the expected grey / white matter share and rare classes, whether GFAP, SNAP25, MBP and CLU are protein IF, and whether the panel will ever be paired with another panel. Until answered:
   - NP8 is `not_applicable`;
   - NR15's protein-IF item uses only channels whose names name a target;
   - the rest is reported as unknown.

   None of these changes a scored row.
4. **NP2's weak or collapsed parents** (CHECK K14). NP2 is recomputed on the production frontal bundle's lookup. Accepting any weak or collapsed parent is the user's call in the gate-P PR, and until then the PR lists them as unaccepted.
5. **§22.9's four questions**, answered before the family is resolved and scored. Until then RESOLVE applies the rules as registered:
   - (a) smoothing emission over neighbouring bins, or reporting the ensemble range. Today the bins are applied as decided, and the class-depth table keeps each bin's member range;
   - (b) the 100-cell per-class minimum of the class-depth prediction. It decides no warning while `real_qc.coverage_min_cells` ≥ 100;
   - (c) `flag_nonneuronal_high_depth` is false for every table cell of the family, because the grid ends at 250 counts;
   - (d) the monotone fill's non-neuronal limit (bins ≥ 1,000) does not reach the open-ended 250 bin, so glia at ≥ 1,000 counts in that bin can be emitted by the fill. A change would be a tightening, but it needs the user's decision.

### 23.11 NP3 readings put to the user, and set a numbers seen before the ruling (M13 NP3 review; recorded 2026-10-06, before the user rules)

**Disclosure.** On 2026-10-06 the review of the NP3 code (M13 chunk C2) scored NP3 read-only on set a's M8 self-map (`$A/m8prep/scratch_store/whb_frontal_supc_clus/resimdrop`). It used the default donor's check half only, `R1_contam_HO` and the bundle's frozen thresholds, with the natural composition approximated by the test cells' type shares. This is not the dry run: no leave-one-donor-out replicate was mapped. At broad (8 classes), the (level, class) records validated were:

| Weighting | Validated | Failing |
|---|---|---|
| unweighted (report-only) | 6 | — |
| test-cell readings (report-only) | 6 | — |
| `natural`, the set reading as scored | 5 | broad Astro |
| `class_balanced` as registered, with the trim and rare-type pooling | 4 | broad Astro, broad Oligo |

Broad Oligo's ≥ 30 set has 277 of 278 confident calls right (.996). Class-balanced it scores .965, below target⁺ .97: the one wrong call is of a rare type, and capped at 10 × the set's median weight it still carries 3.5% of the set's weight. These numbers were seen before the readings below are ruled on. Adopting a reading that validates more classes is therefore a post-hoc loosening (§1 rule 2). It needs the user's written approval, or it goes through §9's revision of an unattainable criterion after the dry run.

**Open** ([P] kept; the code's behaviour until the user rules):
1. **"The reference's natural composition within the class"** (§9 NP3). Neither §23.9 nor §23.10 says how it is read.
   - *Scored:* the set reading. Every truth type of the called class's tested set, a wrong call's type included, takes its share of the reference composition (`natural`).
   - *Reported:* the test-cell reading (`natural_test_cells`), which rebalances the types within each truth class over the scope's test cells.
2. **The class-balanced weighting's trim and rare-type pooling.** §23.9 item 2 (equal total weight per truth type within the called class's tested set) is [U] and scored as written. Two further rules come from the M13 NP3 chunk spec, not from §23.9:
   - the judged-set trim: each weight is capped at `weight_trim_factor` (10) × the set's median positive weight;
   - the pooling of a truth type with fewer than `weight_min_type_cells` (20) calls at its broad class.

   Both move `class_balanced` towards passing, so they belong to the same ruling. Example: 1,000 calls of one bin, 990 right in three types of 330 and 10 wrong in two types of 5 of another broad class. The class-balanced precision is .908 with the trim and .60 without it. Under the literal rule, a set below 60 counts (target⁺ .97) has room for almost no wrong-type call.
3. **Values on a family dataset's composition** (§9 NP3: "reported when one exists"). These are not computed, because that composition needs the dataset's labels, and no real-label input reaches gate P other than NP5's registered profile (§23.9 item 6; CHECK K7). The values on the dataset's depth histogram (label-free) are reported.

### 23.12 NP5 readings put to the user (M13 NP5 review; recorded 2026-10-06, before the set a dry run)

These readings come from the review of the NP5 code (M13 chunk C3). They were found on synthetic tables, and no NP5 output of set a or of the family is recorded. Nothing here changes a threshold, target or limit: the t* range stays ≤ 0.05, the extrapolated share ≤ 0.5 and the compared bins need 50 test cells.

**Open** ([P] kept; the code's behaviour until the user rules). Each is ruled before the set a dry run is scored. Choosing a reading after its numbers are seen, when it validates more classes, is a post-hoc loosening (§1 rule 2).

1. **The median or the profile shares** (§9 NP5: "its cells at the family's expected depth").
   - §23.9 item 6 and §23.10 item 6 (D8, confirmed) make the expected depth each class's median. Plan §8.3 v7.5 says the "> 50% extrapolated" test "uses the class's profile shares".
   - The two readings are not nested. The version-7 monotone fill can mark a bin extrapolated between bins that are emitted on their own verdicts, so the extrapolated bins need not be contiguous, and either reading can fail a class that the other passes:
     - decisions that mark 30, 120 and 250 extrapolated, and a profile of 10% at 20, 20% at 40, 25% at 80, 25% at 150 and 20% at 300 counts: the shares give 0.65 (fail), the median (80, in the 60 bin) passes;
     - decisions that mark only the filled 60 bin, and a profile of 45% at 40, 10% at 70 and 45% at 150 counts: the shares give 0.10 (pass), the median (70, in the 60 bin) fails.
   - *Scored:* both. A class passes only when both are at most 0.5 (`median_share` and `profile_share` are reported side by side). Scoring either alone would pass a class that this rule fails.
2. **A t* range on too few replicates.**
   - A replicate with fewer than 50 fit-half calls in a tested set has no fit and is left out of the range. A set with fewer than two thresholds passes the t* part (`evaluable` false).
   - A tested set that just reaches 200 pooled confident calls can hold fewer than 50 fit-half calls per replicate. The thinnest sets can then pass the t* part on one threshold or none, as NP4's vacuous sets pass NP4 (§23.9 item 3), and such a set counts as a pass in the every-member verdict.
   - The NP5 class table reports, per class, how many tested sets passed this way (`n_tstar_not_evaluable`, with their labels) and how many (set, replicate) pairs were left out (`n_tstar_unfitted`).
   - *Alternative:* fit t* on all of a replicate's calls in the set. That departs from the fit-half rule that `decide` uses.
3. **The version-7 saturated cap counts as a t*.** A set whose fit never reaches the target, but whose fit-half calls are saturated, takes the cap of 0.99 (v7.8), as `decide` applies it. NP5 puts the cap in the range like a fitted t*, so a replicate at the cap beside one at t* 0.90 fails the set (range 0.09).
4. **Other readings**, strict where there is a choice (listed in the `gate_p` module docstring):
   - a bin is compared when the base run or the replicate has 50 test cells there;
   - the emission boundaries are the base run's status changes inside the grid (the grid's edges are none), and at most one compared bin may flip, one that flanks a boundary;
   - a replicate whose fit exists but never reaches the target, and is not capped, fails the t* range;
   - t* is re-fitted in each replicate on its calls of each tested set (the shared membership rule), not read from the replicate's re-derived decisions;
   - a class without cells in the profile takes D8's overall median as one depth, so both its shares are 0 or 1. The pooled profile's shares are the alternative.

### 23.13 NP7 readings put to the user (M13 NP7 review; recorded 2026-10-07, before the set a dry run)

These readings come from the review of the NP7 code (M13 chunk C5). They were found on synthetic tables, and no NP7 output of set a or of the family is recorded. Nothing here changes a threshold or limit: per level, the confident calls to sink or region-implausible nodes stay at most 1% of the level's confident calls (human); at every tested set, one wrong node takes at most 5% of the truth class's confident calls; the region column is `region_plausible_frontal_cortex` (CHECK K4). §23.9 item 5 and §23.10 item 5 (the truth view, the K9.1 denominator, the region) are not reopened.

**Open** ([P] kept; the code's behaviour until the user rules). Each is ruled before the set a dry run is scored. Choosing a reading after its numbers are seen, when it validates more classes, is a post-hoc loosening (§1 rule 2).

**Corrected 2026-10-07, before the user ruled** (second review of the NP7 code, on synthetic tables; no output of set a or of the family read). The first record of this section misstated which reading is the stricter one in its items 3 and 4 (now items 4 and 5), and it did not list the numerator reading (now item 3). Items 3–5 below replace that text, and the code now scores the stricter readings they name. Items 6–9 were items 5–8.

1. **When a call to a sink or region-implausible node is confident.** Such a call has no class (`parent` is null), so it has no frozen threshold of its own.
   - *Scored:* it is confident when the larger of its own bp and the bp of its cell's supercluster row at the same depth reaches the lowest frozen threshold emitted at its level and depth, for any class (K9.1's "emitted bins"). It never counts at a depth where the level emits nothing, or when both bp are missing. The supercluster bp is needed at broad and NT: both WHB sinks (Miscellaneous and Splatter) are Mixed/Unknown at broad, so their calls there name no group and have no bp of their own.
   - *Stricter:* count every call to such a node, whatever its bp. That is how H2 counts (§7 item 1 treats counting only confident implausible calls as a loosening of H2), though §14 NP7 says "confident calls". The NP7 table reports this count (`n_excluded_calls`) beside the scored one.
   - *Not stricter:* the threshold of the cell's truth class at that bin, or the level's raw default threshold.
   - This is the only item where the scored reading is not the strictest one named.
2. **The 1% part's denominator.** The calls to excluded nodes are counted beside the level's confident calls, not added to them: the share is their number over the level's confident calls. Adding them to the denominator is looser.
3. **The 1% part's numerator.** Every confident call to an excluded node at the level counts, whatever the cell's truth class.
   - At NT the denominator holds the neurons' calls only, because a glial call has no NT class. A glial cell assigned a WHB sink still counts in NT's numerator, judged on its supercluster bp (item 1).
   - Example: 1,500 excitatory, 1,500 inhibitory and 3,000 astrocyte cells, and 40 more astrocytes on Splatter. The share is 40 / 6,000 = 0.67% at lineage, broad and supercluster (pass) and 40 / 3,000 = 1.33% at NT. Every NT class fails, although no neuron reaches a sink. NT has rank 2, so this caps `validated_max_level` at broad.
   - *Looser:* count only the cells whose truth class is one of the level's classes, that is, the classes of its frozen decisions (at NT, the neuron classes). The NP7 table reports this count (`n_excluded_level_population`) beside the scored one.
4. **Excluded calls in the single-wrong-node view.**
   - *Scored:* both views, and a tested set passes only when both pass.
     - *With them* (`wrong_node*`): the confident calls to excluded nodes enter the truth view as wrong calls, on the node of item 5. A sink that takes more than 5% of a class therefore fails the class at every level, even when the level stays within 1% (§12 M13: "a planted sink absorbing > 5% of a class fails NP7"). This includes calls that the cells table marks correct because their group matches the truth at lineage, broad or NT (an excitatory cell assigned Amygdala excitatory; at lineage, a neuron assigned a WHB sink), which production never emits.
     - *Without them* (`classed_wrong_node*`): the truth view holds only the class's confident calls that have a class.
   - Neither view is nested in the other. The excluded calls also enter the denominator of the view with them, which lowers every other node's share. Example: class X has 380 right calls, 21 on node nY and 30 calls to excluded nodes spread over 10 sinks. With them nY takes 21 / 431 = 4.87% (pass); without them, 21 / 401 = 5.24% (fail).
   - *Looser:* either view alone. The first record of this item called the view with them the stricter one, which the example contradicts.
5. **The wrong node at the coarse levels.**
   - *Scored:* a wrong call's node is its call at the level (a group at lineage, broad and NT). A call to an excluded node also takes its call where the level names a group, and its assigned supercluster where it names none (the WHB sinks at broad and NT).
   - Merging nodes can only raise the largest share, so this is the strictest of the readings named here. Example: astrocytes with 12 calls on a frontal excitatory node and 10 on Amygdala excitatory. At broad both calls name "Neurons", one node with 22 / 402 = 5.5% (fail).
   - *Looser:*
     - each excluded call on its assigned supercluster. This was the first record's reading: in the example, "Neurons" 12 / 402 = 3.0% and Amygdala excitatory 10 / 402 = 2.5% (pass). It is reported (`split_wrong_node*`).
     - every wrong call on its assigned supercluster, which splits a group's wrong calls over its nodes.
6. **A level over 1% fails every class of the level.** Looser: fail only the classes whose truth cells reach the excluded nodes.
7. **An empty truth view fails.** A tested set whose truth class has no confident call in the set's scope fails (its share is undefined), in either view of item 4. Looser: pass it, as NP4's vacuous sets pass (§23.9 item 3).
8. **A node outside the vocab** counts as region-implausible, as production reads it (`not_in_vocab`, reported apart). A blank `sink` or region flag in the vocab snapshot is refused: the self-map's level specs read a blank sink as false and a blank region flag as true, so no reading of it agrees with both. A cells table built with another region or vocab is refused in both directions: a call to an excluded node that has a class, and a supercluster call to a plausible node with a floor class that has none (where the table's region marks the node implausible, NP7 would otherwise leave its calls out of both sides of the 1% share).
9. **Unweighted shares.** NP7's shares are unweighted, as NP4's are (§23.9 item 3), because §14 reweights NP3 only. The reweighted shares are not computed.

### 23.14 NP4 seed readings put to the user (M13 NP4 seed-criterion review; recorded 2026-10-07, before the set a dry run)

These readings come from the implementation and review of NP4's seed criterion (M13 chunk C6, `gate_p.np4_seed_stability` and `np4_class_verdicts`). They were found on synthetic tables. No replicate of set a or of the family has been mapped at seed 1, so no output of the criterion exists. Nothing here changes a threshold or limit: seed 0 vs 1 changes at most 2% of confident labels per validated level (§9 NP4) [R]. §23.9 item 3 and §23.10 item 3 are not reopened.

**Settled by D6** (§23.9 item 3, confirmed in §23.10). The seeds are MapMyCells mapping seeds. Seed 1 re-maps each group's seed-0 simulated cells, and a label changes when seed 1 gives the same simulated cell (level, cell, depth) another call, whether that call is confident or not, a sink call included. This is the statistic of `resolvability.seed_stability`, and a seeded test checks that the two give identical shares. It follows that:
- a label that keeps its name and falls below its threshold at seed 1 is a threshold crossing. It is reported (`n_crossed`) and not counted.
  - D6 gives the reason: near the thresholds the bootstrap probability is noisy (a standard error of about 0.045 at 100 iterations, §20 D12 (b)), so counting crossings would fail by design.
  - Gate H's figures of 7.0% (HD1) and 13.4% (HD2) at M8 (§20 D12) illustrate that noise, but they are not this statistic. They come from real sections, they also count cells confident at seed 1 only, and they include seed-0 confident labels whose seed-1 call is another, unconfident name, which gate P counts as changes.
  - They are therefore neither an estimate of gate P's `n_crossed` nor evidence that the D6 statistic stays under 2%.
  - *Corrected 2026-10-07* after the M13 NP4 seed-criterion review. The first wording cited them as crossings of this criterion;
- a change to a name that is confident at seed 1 is also reported on its own (`n_switched`). This is gate H's reading of H15 (§20 D12 (a)). It is looser, because it ignores changes to a call that is not confident;
- gate P stops with an error unless seed 1's table holds the same simulated cells with the same `total_counts` as seed 0's. A simulation seed changes the counts, so it is refused rather than scored.

**Open** ([P] kept; the code's behaviour until the user rules). Each is ruled before the set a dry run is scored. Choosing a reading after its numbers are seen, when it validates more classes, is a post-hoc loosening (§1 rule 2).

1. **The level's share, not each class's** (§9 NP4: "per validated level"). This reading needs the user's explicit answer.
   - *Scored:* the share over all the level's labels that are confident at the frozen thresholds, over every class and emitted bin of the level. This is the population of `seed_stability` and of NP7's 1% denominator (CHECK K9 item 1). Classes outside C_P and classes without a tested set count. A level over 2% fails NP4 for every class of the level, the classes without a changed label included.
   - One unstable class can therefore fail every validated class of its level. §12 M13's test "one failing class leaves the other classes of its level validated" then holds for NP4's tested-set parts (the floor and the range rule), not for the seed criterion. This departs from NP4's per-(level, class) design.
   - *Alternative:* each class's own share of its labels. The two readings are not nested. Example: one group's broad level holds 10,000 confident labels.
     - If 250 of class Z's 1,000 labels change and nothing else does, the level's share is 2.5%, and every broad class fails. Each class's own share is 0%, except Z's 25%.
     - If instead 10 of class W's 50 labels change, the level's share is 0.1%, and every class passes. W's own share is 20%.
2. **Each group is scored.** Every group's seed pair must change at most 2% of the group's labels at the level. The share pooled over the groups (`pooled_changed_share`) is reported only. *Looser:* the pooled share alone, which can pass when one group is above 2%.
3. **The denominator is the seed-0 labels**, as in `seed_stability`. A cell that is confident at seed 1 only is not counted. *Alternative:* the labels that are confident at either seed, with a cell counted as changed when its two calls differ. Neither reading is nested in the other.
4. **A group without confident labels at a level passes it**, because its share is undefined (`changed_share` is `nan`). NP4's vacuous sets pass in the same way (§23.9 item 3). *Stricter:* fail it.
5. **A class that the WHB COP rule drops at seed 1 only.** `whb_cop_rule` nulls the broad `parent` of a cell whose supercluster COP call fails the rule, and keeps its call. When this happens at seed 1 only, the seed-0 label keeps its name. It is therefore a threshold crossing (`n_crossed`), not a change, as in `seed_stability`, which compares calls only, although production gives that cell no broad label at seed 1. *Stricter:* count it as a change.

### 23.15 NP6 readings put to the user (M13 NP6 implementation; recorded 2026-10-07, before the set a dry run)

These readings come from the implementation of NP6 (M13 chunk C4: `resolvability.gate_p_stress_members`, `gate_p.np6_set_stats`, `np6_verdicts`, `np6_class_verdicts` and the cross-platform stress asset). They were found on synthetic tables. No stress recipe has been simulated or mapped on set a or on the family, so no NP6 output exists. Nothing here changes a threshold or limit: at each tested set the stressed calls reach point precision ≥ target_L and a Wilson bound ≥ target_L − 0.02, their drop against the base is not significantly above 0.05 (one-sided 95%), and a set needs 200 stressed calls before it is scored on its own (§9 NP6) [R]; the stresses are spill 0.35, LogNormal(0, 1.0) and the offsets capped at ±2 log2 (plan §3.7 `gate_p_stress`) [R]. §23.9 item 4 and §23.10 item 4 (the drop test and D7's factors) are not reopened.

**Settled** (§9 NP6 and "Version-7 families", D7, D12, CHECK K8; listed so that the record is complete):
- *The stress recipes.* Each R1 emission member is stressed three times: spill 0.35, LogNormal(0, 1.0) and the cross-platform offsets. Each R3 member is stressed at spill 0.35 only, and human Xenium Prime families add `R1_xtissue_lung_stress` (K8; how many times is item 7 below). A version-6 family's base is R1@0. Each stress keeps its member's seed and changes one setting; the LogNormal stress scales the member's own normals by 1.0 / 0.8. This family (R1 × 8, no R3, no lung table) has 24 stress members.
- *The factors* (D7 (a) + (i)). The asset `ratio__xenium_v1_vs_merscope__human_brain_ffpe` holds set a's 296 genes from the registered D-A7 basis, centred on the median gene and capped at ±2 log2 (how the four pairs are combined is item 8 below). Its provenance records the source path and sha256 under the evidence root and the deriving script (`scripts/annotation/build_xplatform_stress.py`) with its sha256; an asset without them is refused. A panel gene the table does not cover takes a keyed resample (seed, gene) of the capped values. The stressed efficiency is the member's draw times 2^offset, normalised to the panel's median gene.
- *X1* (D7 (b)): the X1 factor table is reported beside the offsets per panel gene (`gate_p.np6_factor_report`: the offset each gene receives, the mean X1 factor per platform, their difference and the genes at the X1 cap). It reads no cell, maps nothing and decides no verdict.
- *Tested sets* are NP3's, built on the base's pooled held-out calls. A set built on other calls (a plain concat of the tables keeps the default donor's fit half) is refused, as in NP3 and NP7.

**Open** ([P] kept; the code's behaviour until the user rules). Each is ruled before the set a dry run is scored. Choosing a reading after its numbers are seen, when it validates more classes, is a post-hoc loosening (§1 rule 2).

1. **The weightings scored.** D12 tests the drop "on each set's Kish n", which differs from n only on a reweighted set, while CHECK K9 item 2 notes that §9 reweights NP3 only.
   - *Scored:* each set unweighted and under NP3's two weightings (natural and class-balanced, each weighing its own calls, with NP3's trim and rare-type pooling), and the set passes only when it passes under all three. The §23.11 ruling on NP3's weightings applies to NP6's weighted rows too.
   - *Looser:* any one weighting alone. Example at broad (target .90): a set of 180 type-A and 20 type-B calls, all right in the base and 171 and 17 right under stress. Unweighted, .94 on 200 calls (Wilson .898) passes; class-balanced, .90 on a Kish n of 72 (Wilson .809 < .88) fails. Neither drop is significant (lower bounds .032 and .042).
2. **How thin sets pool, and where a pooled test cell is scored** (§9: "those left with < n_min stressed calls are pooled with the next deeper set").
   - *Order:* a (level, class)'s tested sets are ordered by their shallowest bin, then their deepest. A set with fewer than 200 stressed confident calls (unweighted) is pooled with the next set in that order, then the next, until it holds 200. The base, stressed and clean values are read on the same pooled sets, and the row stays the original set's, so every tested set gets its own verdict. A bin at D_P tested on its own therefore pools with the "≥ D_P" set. A bin above D_P tested on its own (between D_P and the deepest bin) comes after "≥ D_P" in this order and has no deeper set when it is the deepest such bin; it is scored on its own calls, and "≥ D_P" covers its cells as well.
   - *Each test cell counts once* in the pooled sets. Every simulation (version 6 `thin_and_contaminate` and version 7) places each test cell at every grid depth its native counts reach, so the cells of a deeper set are also cells of each shallower set (they are nested). Which of a cell's rows is kept therefore decides what a thin set is scored on.
   - *Scored (the stricter reading):* each test cell is kept at its row in the thin set; a deeper set adds only the cells the thin set lacks. With nested cells, pooling adds no cell, so a thin set is scored on its own stressed calls and item 3 applies (its Wilson bound on few calls decides).
   - *Looser:* each test cell is kept at its deepest row in the pooled sets (the rule of a "≥ D_P" set, §9 "each test cell counted once at its deepest bin", applied across the pooled sets). A thin set is then scored on the deeper set's rows of the same cells, and its own stressed calls are not judged; only their number is reported (`n_stressed_tested`). Example at broad (target .90): 300 test cells reach 60 counts, and the base calls all of them confidently and correctly at 30 and at 60 (two sets tested on their own). A stress that keeps 210 confident calls at 30 at precision .50 fails "30" under both readings (210 ≥ 200, so "30" is scored on its own). A worse stress that keeps only 150 confident calls at 30 at precision .33 leaves "30" thin, so it pools with "60". Under the scored reading "30" is scored on its 150 calls at .33 and fails. Under the looser reading it is scored on the same cells' 60 rows (300 calls at 1.00) and passes. Shallow bins are where the LogNormal and offset stresses are expected to remove confident calls, so the case is likely for smaller classes.
   - The code keeps both (`Np6Settings.pool_rows`: `first_set`, the default, or `deepest_row`); tests pin each on nested and non-nested cells.
3. **A set still thin when nothing deeper is left** is scored on the calls it has, so its Wilson bound on few calls decides (reported in `below_min_confident_n`). A set without any stressed confident call fails, because its precision is undefined. *Looser:* pass such a set, as NP4's vacuous sets pass (§23.9 item 3), or mark it not evaluable.
4. **Each simulation fills a set's scope itself.** The stressed calls of a tested set are the stressed simulation's own rows at the set's level, class and depths.
   - On the version-7 grid of total counts, spill 0.35 lowers the host target from D / 1.25 to D / 1.35, so a stressed bin can hold test cells that the base bin does not, and a test cell's deepest row can differ between the two. On the version-6 grid the hosts are the same.
   - The per-cell thinning, partner and spill draws are keyed by the recipe's name (version 5), so the stressed simulation is an independent draw of the same test cells. That is what D12's unpaired two-proportion test assumes.
   - *Alternative:* score only the stressed calls of the base's test cells of each set. Neither reading is nested in the other.
5. **The direction of the offsets.** The asset holds log2(Xenium / MERSCOPE), and every family's R1 draw is multiplied by 2^offset whatever its platform, so a gene that reads higher on Xenium is raised. *Alternatives:* the opposite sign for a MERSCOPE family, or both signs as two stress recipes (stricter, with twice the stress compute of this recipe). For this family's measured genes the sign decides which genes lose counts; for the resampled genes it only mirrors the pool.
6. **A mouse family** takes the human offsets too, with every gene resampled from them, because §9 names "the measured human cross-platform offsets" for every family. Mouse gate P waits on M6b (C20–C22), so this is recorded to be ruled before a mouse family is scored.
7. **How often a human Prime family takes the lung stress.** §8.3 v7.4 defines `R1_xtissue_lung_stress` on R1@0 only ("the R1@0 efficiency multiplied by the ratio, other genes by a keyed LogNormal(0, 0.702)"), and §9 "Version-7 families" says only that human Prime families "add `R1_xtissue_lung_stress` to NP6's stress recipes".
   - *Scored (the stricter reading):* the lung stress is added once per R1 emission member, at that member's seed: R1@s's efficiency times the ratio, and the keyed LogNormal at s for the genes the ratio does not cover. Each is compared with its own base member, as the other R1 stresses are. A Prime 5K human family (R1 × 8) has eight lung stresses.
   - *Looser:* one lung stress at R1@0, compared with R1@0 only (one stress instead of eight). This family has no lung table and is not affected; the reading binds the human Prime 5K gate-P run that joins this pass (D4).
8. **How the four pairs of the cross-platform offsets are combined.** D7 and §23.10 name the source table and say that the offsets are centred on the median gene and capped at ±2 log2. Neither says how the four pairs' columns are combined.
   - *Implemented:* each pair's column is centred on its median gene (the pair medians, −0.12 to +1.52 log2, are a platform-wide scale that the median normalisation of every efficiency removes anyway), the four centred values are averaged per gene, and the means are centred on the median gene and capped (`scripts/annotation/build_xplatform_stress.py`).
   - *Effect on the committed asset (296 genes):* the averaged offsets have an SD of 1.617 log2 before the cap, against 1.59–1.77 for the single pairs, and 20.6% of genes are capped, against 19–24%. So the averaging barely narrows the spread, and the stress is about as strong as any one pair's.
   - *Alternatives:* the median of the four centred values (SD 1.26 after the cap and 19.9% capped, against 1.27 and 20.6% for the mean); or one stress recipe per pair (stricter, at four times the compute of this recipe).

### 23.16 Assembly readings put to the user (M13 chunk C7; recorded 2026-10-07, before the set a dry run)

These readings come from the implementation of gate P's assembly (M13 chunk C7): `gate_p.gate_p_class_sets`, `gate_p_class_records`, `gate_p_validated_max_level` and `assemble_gate_p`; the family checks `np1_gene_ids`, `np2_panel_coverage`, `np8_cross_panel` and `np9_resources`; the NP1–NP9 report; and `diagnostics.write_simulation_family`. They were found on synthetic tables. No gate-P output of set a or of the family exists. Nothing here changes a threshold or limit, all of them [R]:
- C_P holds the classes with ≥ 700 pooled test cells, and it must hold ≥ 90% of them;
- a (level, class) is validated when NP3–NP7 pass in every emission member;
- `validated_max_level` is the deepest level at which every class of C_P is validated, and promotion needs at least broad (human) or class (mouse);
- NP1: ≥ 95% resolved (≥ 98% with vendor IDs);
- NP2: ≥ 10 root markers, and ≥ 10 markers for each root child with n ≥ 50;
- NP9: within 1.5× of the time reference.

§23.9 items 1, 9, 10 and 11 and the decisions of §23.10 (D10, D13, D14, D16, D29) are not reopened.

**Settled** (listed so that the record is complete):
- **D13 (a).** The writer refuses a family whose bundle has no resolvability self-map.
- **D14 (a).** No NT row is written for a class where NT does not apply, and no SEA-AD subclass row. The writer refuses both, and any class that is not a consensus class key of its level.
- **D16.** The writer refuses a `family_id` other than the panel's own hash-derived id.
- **D29.** An NP4 failure at any tested set fails the (level, class).
- **D10 (a).** The time reference for version 7 is `gate_p.np9_time_reference_v7`.
- **NP8** is `not_applicable` until the user says the family will be paired (D26).
- **NP2's weak and collapsed parents** (§23.10, still open item 4). NP2 is `pending` until the user accepts them in the gate-P PR, and a pending NP2 does not pass.
- **Writing the rows.** Validated rows are written only for a family that passes gate P. They are appended to the packaged tables with the existing lines kept, and are read back through every check of `read_validated_panels` before they replace the files. `validated_panels.csv` is replaced last, and a failed replacement restores the files already replaced.
- **D2.** C_P is fixed on disjoint replicates (D2 (d)): a test cell in two groups stops gate P, the default donor's fit half included. If the user chooses the fallback (c) (§23.10, still open item 2), `gate_p_class_sets` takes `disjoint=False`: each shared cell is counted once, and the default donor's fit-half cells are left out of every group, as §23.2 defines (c).
- **NP9's members.** NP9 records the members its re-runs were scored on, and the assembly refuses an NP9 whose members are not the verdicts' emission members (CHECK K13).

**Open** ([P] kept; the code's behaviour until the user rules). Each is ruled before the set a dry run is scored. Choosing a reading after its numbers are seen, when it validates more classes, is a post-hoc loosening (§1 rule 2).

1. **C_P's denominator.** Some pooled test cells have no class at a level: sink truths such as WHB Splatter and Miscellaneous, whose `truth_parent` is null.
   - *Scored (stricter):* they stay in the denominator of the 90% rule.
   - *Looser:* the share over classed cells only, as `resolvability.gate_p_class_set` computes it. It is reported as `class_set_share_classed`.
2. **The NT population.** At NT, the cells whose truth is `not_neuron` are left out of NT's counts and of its denominator. These are the non-neuronal cells and the sink cells. NT's C_P and its 90% rule are therefore read on the neurons.
   - *Scored (looser than the literal reading):* this follows D14 (a).
   - *Literal (stricter):* §14's "≥ 90% of the pooled test cells" counts those cells too. That would put NT below 90% on any panel and cap every family at broad.
3. **The 90% rule decides the level.** §14 says only that C_P "must hold" ≥ 90% of the pooled test cells.
   - *Scored:* a level whose C_P holds less than that is incomplete, so neither it nor any finer level can be `validated_max_level`.
   - *Looser:* report the share only.
4. **The version-7 depths.** For a (level, class) validated in every member:
   - `validated_min_depth` is the deepest of the members' NP3 values, so the class is validated only where every member validates it;
   - `tested_max_depth` is the shallowest member D_P, so a label is extrapolated wherever any member extrapolates it. It is raised to `validated_min_depth` when it lies below that.

   Each member's values are reported. `tested_max_depth` only marks labels as extrapolated; it decides no emission and no validation mark. *Alternatives:* the deepest D_P, or the median.
5. **Status precedence.** A (level, class) that fails any criterion in any member is `failed:NP<k>`, with k the lowest failing criterion, even when another criterion is not evaluable. The record lists every failing criterion and member. *Looser:* `not_evaluable` before `failed`.
6. **Classes outside C_P** get records with `in_class_set` false and their own status. A validated one is marked `ct_<L>_validated` in production. They never decide `validated_max_level`.
7. **NP1.**
   - *Vendor-supplied IDs.* The 98% bar applies when at least one feature of a declared panel resolved from the vendor's own IDs (the `native` source). The family's run resolved 491 of its 496 genes that way (§23.10), so its bar is 98%. *Looser:* apply the bar only when every feature carries a vendor ID.
   - *Controls reaching the query.* Every platform's anchored control-name rule is applied to the symbols of the annotation panel. The substring rule is not, because a resolved gene can contain a control token. *Stricter:* apply both rules.
   - *The unresolved list.* While it is not empty and the user has not reviewed it in the gate-P PR, NP1 is `pending` and does not pass. The family's list is empty (§23.10).
   - *The species test.* A test that was not run, or that is `not_evaluable`, leaves NP1 not evaluable, so the family does not pass.
8. **NP2's root children.**
   - *n* of a root child is its cells in the production primary bundle's reference. The gate-P driver supplies it, with the child's markers from the bundle's lookup (CHECK K14).
   - A child without children of its own counts with the root's markers, as `reference.root_children_with_markers` counts it.
   - *Alternative:* the child's pooled test cells.
9. **NP9.**
   - *Prefilter scope.* Above 1,000 genes, the part is scored on `simulate.prefilter_verdict`, which judges every class with ≥ 50 calls at every emitted level. That is a superset of §14's "validated level and class", so the reading is stricter. A bundle whose markers were not prefiltered has nothing to compare, and the part is `not_applicable`.
   - *The measured time* is PREP, resolvability included, plus the gate-P replicates. The limit is 1.5 × the reference. For version 7 the reference is D10 (a)'s: the set a version-7 dry run's time × the family's simulated cells / the dry run's simulated cells.
   - *Missing measurements.* A part without its measurement is `not_evaluable`: no time, no RSS, or a member without a re-run (CHECK K13's scope: PREP's bundle and one seed-0 replicate per member). The family then does not pass.

### 23.17 Gate-P driver readings put to the user (M13 chunk C8; recorded 2026-10-07, before the set a dry run)

These readings come from the gate-P driver (M13 chunk C8: `gate_p_run.run_gate_p`, registered by `annotation-panel-simulate --gate-p` and `scripts/acceptance/new_panel.py`), from the leave-one-donor-out sets of `reference.build_whb_frontal_ho` and from the per-donor pool report `reference.ho_donor_pool_sizes`. They were found on the synthetic WHB fixtures of the tests. No gate-P run has been made on set a or on the family, so no gate-P output exists. Nothing here changes a threshold, a target or a criterion of §9 or §23.9.

**Settled** (§23.9, §23.10 and the M13 CHECK; listed so that the record is complete):
- *D1 (a).* Every leave-one-donor-out set leaves its other-region COP cells out of what it simulates, as the default donor's does (`reference.self_map_test_cells`); each set's exclusion is recorded in `gate_p_run.json`.
- *D2 (d).* Each extra donor's held-out bundle is built with `resolvability.holdout_donor` set to that donor, its other-region top-up (and the version-7 class top-up's other-region pool) drawn from that donor's own cells only, and none of the default held-out test set's cells (every cell of it, the D1-dropped ones included). Both settings enter that bundle's `build_hash`; every production payload, bundle and `build_hash` is unchanged. A leave-one-donor-out set that holds a default test cell stops gate P.
- *D2 (c), the fallback.* With `--gate-p-other-region shared` the extra donors' bundles take the production other-region draw. The default donor's fit-half cells leave every other group. In the pooled sets a cell held by several groups is kept once, at the row of the first group that holds it, the default donor first and then the donors in name order. C_P counts it once (`disjoint=False`).
- *D4.* `--gate-p` needs `--species`, never taken from a public panel's record. A mouse family is refused before any compute until M13 C20–C21 (the second WMB draw and the MS1 / MS2 dry run).
- *D5 (c) and CHECK K11.* `--gate-p-version 7` scores a version-6 family's version-7 ensemble: its emission members are simulated on the bundle's own test set and engine, as `simulate.run_v7_diagnostic` does, and nothing is written to a store.
- *D11 (b).* Gate P builds its bundles in the store the command is given, and writes nothing inside a store. A store with a root that is, or lies inside, the config's `reference_store` or `reference_store_large` (the production stores) is refused before any compute, so `--store` and `--store-large` must both name the separate gate-P store.
- *CHECK K13.* The identity re-runs are one seed-0 replicate per emission member (the default donor's, compared with PREP's stored rows) and PREP's bundle (rebuilt in a scratch store; its self-map tables, decisions, lookup, mapping tree, vocab snapshot and held-out test cells are compared).
- *The realisation.* Every replicate is mapped with the self-map's recorded MapMyCells worker count (§18 item 2). Another count is refused after PREP and before any gate-P build or mapping. A bundle that PREP builds in the run is mapped with the run's own count, so only a bundle reused from the store can differ, and reusing it costs no compute.
- *When a run is refused* (corrected 2026-10-07 after the C8 review). Before any compute (`gate_p_run.precheck_gate_p`, which `run_panel_simulation` calls before the panel is computed or a bundle is built), gate P refuses: another species; a production store; an output inside a store or one holding an earlier run or its replicates; configured donors that repeat, number fewer than two, or leave out a fixed default donor; seeds without 0 and another seed; and a missing NP5 depth source. After PREP and before any gate-P build or mapping, it refuses what needs the built bundle: no self-map onto the held-out bundle; donors that are not frontal donors of it; a default donor not among them; another worker count; and a held-out spec whose `build_hash` differs. The simulation report is written before gate P runs, and such a refusal is recorded in it under `gate_p` (`refused`) before it is raised.

**Open** ([P] kept; the code's behaviour until the user rules). Each is ruled before the set a dry run is scored. Choosing a reading after its numbers are seen, when it validates more classes, is a post-hoc loosening (§1 rule 2).

1. **What "a donor's own pool cannot meet the per-class top-up rule" means** (§23.10 open item 2).
   - *Implemented:* for each extra donor and each non-neuronal class that the default held-out test set holds, the donor's own frontal test cells plus its own admissible other-region cells (of its own leave-one-donor-out training reference, none of the default test set) must reach `topup_min_class_test_cells` (200, the class minimum of §21.3). Other-region cells that the self-map then drops (D1 (a): COP) do not count; the donor's own dropped cells are listed in their own column (`n_other_region_own_dropped`). Under D2 (d) one short class stops gate P before any replicate is mapped, unless the user chooses (d) as it stands (`--gate-p-accept-small-pools`) or the fallback (c). The report lists every donor and class with its frontal, own other-region and every-donor other-region cells and the default test set's count.
   - *Alternatives:* compare each donor with the default test set's count of the class, or stop only on classes of C_P. Either stops less often.
2. **C_P's test cells.** C_P is fixed from each group's test-cell table, before any cell is mapped: every test cell after the D1 drop (the default donor's check half), each with its truth class per level. Test cells whose native counts reach no grid depth are counted although they are never simulated. *Alternative:* count only the cells some replicate simulates.
3. **NP5 in a version-7 family.** Each emission member's decisions are re-derived in each replicate and compared with that member's own base decisions (the default donor at seed 0). The ensemble's re-derivation, compared with the frozen ensemble decisions, is reported beside them (`np5_ensemble_agreement`) and not scored. *Stricter:* score both.
4. **NP9's measured time and the version-7 reference.** The measured time is PREP, read from the `bundle.json` timings of the primary bundle and its held-out test set, plus this run's leave-one-donor-out builds and replicates; the identity re-runs are left out. The version-7 reference (D10 (a)) scales the set a dry run's `measured_seconds` by the family's `simulated_cells` over the dry run's `simulated_cells` (both in `gate_p_run.json`). `simulated_cells` counts the simulated cells of every replicate gate P scores, including the default donor's seed-0 emission and clean rows. In a version-7 family PREP simulated those rows (their count comes from the self-map summary, their time is in PREP's). In the forced version-7 dry run, gate P simulates them itself. Both runs therefore count on one basis; `gate_p_run.json` gives the two parts (`simulated_cells_this_run`, `simulated_cells_prep_rows`). Corrected 2026-10-07 after the C8 review: the count first left out PREP's rows in a version-7 family but not in the dry run, which put the family's reference too low by the share of those rows, (E + 1) / (3 (5E + 1)) for E emission members with three donors, two seeds and three stress members (about 7% for eight), a tightening. In both runs, PREP's other simulations (those gate P does not score) stay in the time and out of the count. A version-7 family's reference is D10 (a)'s only. A stated version-6 reference (`--gate-p-time-reference-seconds`) is not used for one, so without the dry run's values the time part is `not_evaluable`. *Alternatives:* include the identity re-runs, or scale by every cell PREP simulated.
5. **NP3's natural composition** is the WHB frontal reference's share of cells per supercluster, over every frontal donor (the held-out test set's region metadata), and the same shares give the excluded classes' reference shares in C_P's report. *Alternative:* each donor's own training reference.
6. **The dry run (D28).** H18's classes are the supercluster classes with ≥ 50 cells of the default donor's held-out test set after the D1 drop (both halves). Supercluster COP is exempt (§18 C1). M8 D4's exceptions are bins of dataset-reweighted runs (broad OPC at 15 and 120 counts, Immune at 60 on one dataset), so they exempt no class. The broad level must also be complete, its C_P holding ≥ 90% of its test cells. *Looser:* treat D4's bins as class exemptions, or drop the 90% condition.
7. **NP5's expected depth source.** A registered per-class `profile` asset (`--depth-profile-asset`, D8 with CHECK K7) gives each class's depths, and the profile's overall median gives every other class's. Otherwise `--expected-depth`, the label-free pooled median, is used for every class, or a pooled profile's median. A per-class CSV is never read for NP5. Gate P refuses to run without one of these, before any compute.

### 23.18 Status before the set a dry run (recorded 2026-10-07, after the M13 chunks were merged on the M13 branch; before the dry run runs)

**What exists.**
- The gate-P code: the criteria NP3–NP7 (M13 chunks C1–C6), the assembly, the family checks, the report and the writer (C7), and the driver (C8). It was tested on synthetic tables only.
- The real-data QC in human RESOLVE (C12–C16).
- The regression of the real-data QC on the M8 human pairs (C17).

No gate-P output of set a or of the family exists. None of the family's outputs has been read since §23.10: the reading rule still holds while D17 is open. This subsection changes no threshold, target or criterion.

**Superseded wording.** §23.9 says the NP4 code "marks its D12 part as pending". D12 was confirmed on 2026-10-06 (§23.10), and the code no longer marks it pending (M13 chunk C2).

**The C17 regression on set a** (2026-10-07; `$A/m13/regression/REPORT.txt`; code `105ace2`; M8 RESOLVE outputs read only). It covers set a's 16 section × segmentation rows (HD1, HD2, HD3 and HD4, both platforms, proseg_hybrid and reseg). It is recorded here before any real-data QC output of the family is read, as D18 and D23 require. The rows are warn-only, because gate H has not merged into `main` (D20 (b)).
- **NR1.** No downgrade-only violation. With and without the QC, the label tables are identical to M8's.
- **NR9, registration G1 (D23).**
  - The fail rule fired on no row, so there is no false failure.
  - The warning rule fired on HD4_MERSCOPE (proseg_hybrid and reseg), with density ratios 1.99 and 1.91, and on HD2_MERSCOPE reseg, at 1.74.
  - Under D23 (b) the move to `failed` + `exclude_hard` (`real_qc.registration_g1_effect = gate_failed`) is decided on this result. The move is a tightening. **Open:** the user's decision, recorded here before any real-data QC output of the family is read. Until then G1 stays warn-only.
- **NR5, the marker referee (D18).**
  - The statistic as implemented (`real_qc.marker_referee_comparator = node`, mouse G2's rule) is `not_evaluable` on all 16 rows. Only one broad class, Microglia with 19 markers, has ≥ 3 derived markers on set a's panel, and the rule needs two classes.
  - The `class` comparator is reported, not adopted. It gives 0.745–0.979 on proseg_hybrid, with HD1_XENIUM at 0.745, below 0.75, and 0.889–1.000 on reseg.
  - By D18, a set a value below 0.75 sends the thresholds back to the user. A `not_evaluable` statistic is not a pass either.
  - **Open:** the user decides NR5 for this family before any number of the family is seen. The options are:
    - keep the `node` statistic, under which NR5 is likely `not_evaluable` on the family too;
    - adopt the `class` comparator, whose set a values are now known;
    - change the marker sets or the thresholds.
  - Each option is a change after set a's numbers were seen, so the user's written decision is quoted in the family's gate-P PR.

**Open before the set a dry run is scored** ([P] kept; the code's behaviour until the user rules). Choosing a reading after its numbers are seen, when it validates more classes, is a post-hoc loosening (§1 rule 2).

1. The NP3 readings, §23.11 items 1–3.
   - These include the trim and the rare-type pooling of the class-balanced weighting. Both come from the M13 chunk spec, not from §23.9, and both move it towards passing.
   - On set a's self-map, `class_balanced` as scored fails broad Astro and broad Oligo (§23.11).
2. The NP5 readings, §23.12 items 1–4.
3. The NP7 readings, §23.13 items 1–9. Item 1, which counts only confident calls to sink or implausible nodes, is the one NP7 reading where the code does not take the strictest option named.
4. The NP4 seed readings, §23.14 items 1–5. Item 1, the level's share against each class's share, needs the user's explicit answer.
5. The NP6 readings, §23.15 items 1–8.
6. The assembly readings, §23.16 items 1–9. Item 2, the NT population, is scored on the reading that is looser than the literal one. If the user keeps it, the user's written approval is quoted in the gate-P PR.
7. The driver readings, §23.17 items 1–7.
8. **NP5's expected depth for the set a dry run** [P]. D8 (§23.10) defines it for the family only. Proposal: the label-free pooled median of `total_counts` over the table cells of the set a family's M8 sections on the scored segmentation (proseg_hybrid).
   - The sections are HD1, HD2 and HD3 on both platforms, and HD4_XENIUM, whose 298-gene panel inherits set a. HD4_MERSCOPE is its own family.
   - The median is read from the M8 RESOLVE label tables, using only their `total_counts` and `in_table` columns (`$A/m13/dryrun/inputs/set_a_expected_depth.py`), and is recorded here before the dry run is scored.
   - *Alternatives:* each platform's own median, or the MERSCOPE sections only. No per-class profile asset exists for set a.
9. **The version-7 dry run's measured time** (NP9's reference for the family, D10 (a)) [P]. The version-7 dry run builds its own leave-one-donor-out bundles in a store of its own. Its measured time then counts those builds, as the family's run will.
   - *Alternative:* reuse the version-6 dry run's bundles. The reference is then lower by the build time, about 5 of roughly 150–240 minutes, which is a tightening for the family.

**NR7's depth matching** (M13 chunk C16; found on synthetic tables, `$A/m13/c16/open1_interpolation_probe.out`; no real data). The gene-complexity check compares the native cells of each grid bin [D_i, D_i+1) with the simulated cells at D_i, the bin's lower edge.
- Native cells in a bin carry more counts than D_i, so they carry more genes even without any complexity gap.
- On synthetic cells with no true gap (native medians of 150–330 counts), the measured gap is 0.18–0.57 per bin, above the 0.45 limit in the shallow bins.
- Interpolating each test cell's simulated genes to the native bin's median total brings the gap to −0.06 to −0.01. This is possible inside the grid but not above its last value.
- NR7 only warns, and it is `not_evaluable` on the family's bundle of 2026-10-02.
- **Open:** the user chooses the reading before any real-data QC output of the family is read: the lower edge (as implemented), interpolation at the native median total, or another matching.
  - Interpolation fires fewer warnings. Whether it is §8.8's "matched depth" or a loosening is part of that ruling.
  - A related reading of the same chunk: the 50-cell minimum counts test cells, not member rows (the docs of the gene-complexity check). It is named in the family's gate-P PR.

**Still open beyond the dry run** (§23.10): D17, the declared panel; D2's "too small", decided with the pool sizes of the run in view; D26's unanswered metadata; NP2's weak or collapsed parents; and §22.9's four questions. The NP5 depth profile of the family (D8 with CHECK K7) does not exist yet. Its asset is written, frozen and recorded only after D17 is frozen, from the family's first provisional `map_first` run with RESOLVE's version-7 consumer.

### 23.19 Decisions of 2026-10-07 on the open rulings (recorded before the set a dry run runs)

The orchestrator put every open ruling of §22.9 and §23.11–§23.18 and the open rows of the decision table to the user in one memo: 63 items, each with options, what the code does now, a recommendation and a label (tightening / neutral / LOOSENING / POST-HOC). An adversarial CHECK section (CK1–CK21) followed. The memo, frozen as answered, is `$A/m13/M13_DECISION_MEMO.answered_20261007.md`.

**The user's answer** (chat, 2026-10-07): "Accept all recommendations, allowing loosenings for A2, A31, C3, A8, A21 and D3 (b)".

- **Every recommendation of the memo is adopted**, as corrected by its CHECK section. 57 of the 63 keep what the code did at `857d75c`. The explicit answers the CHECK asked for follow the recommendations:
  - **A17 (a):** NP4's seed limit "≤ 2% per validated level" is the level's share, over every class and emitted bin.
  - **C1 (b):** the marker referee (NR5) uses the `class` comparator, with the registered 0.75 / 0.70 thresholds unchanged. This is a tightening, and POST-HOC: the registered `node` statistic was not evaluable on set a.
  - **C2:** registration G1 moves to `gate_failed`. A tightening.
  - **A48 (b):** if the pool-size stop fires, the shared other-region draw (D2 (c)) is used, the same for the v6 and v7 dry runs and the family. Per CK17, this replaces the registered "decide with the sizes in view" step. The pool sizes are reference metadata and are still reported.
  - **CK1 (a):** a class H18 expects that gate P cannot evaluate (fewer than n_min = 200 confident calls) fails the dry run, as coded.
  - **CK2:** after a criteria revision (§9), the dry run is re-run under the revised criteria and must pass. A failed dry run is never re-scored under another reading.
- **Loosenings, approved in writing by the user:**
  - A2 (a): keep the class-balanced weight cap (10 × the set's median) and rare-type pooling (< 20 calls). This one is POST-HOC.
  - A31 (a): NT's C_P and its 90% rule are read on neurons only.
  - C3 (b): NR7 matches each native depth bin to simulated cells by interpolation, not at the bin's lower edge.
  - A8 (a): a call to a sink or region-implausible node counts as confident when the larger of its own and its supercluster's bp reaches the lowest frozen threshold emitted at that level and depth.
  - A21 (a): a WHB COP-rule drop at one seed only is a threshold crossing.
- **D3 (b), amending text confirmed on 2026-10-06:** M13 closes on the human path. The mouse chunks C20–C22 (the mouse dry run, the mouse regression rows and the mouse per-donor spread) move into the mouse work after M6b. They still run before any mouse family is scored (D4: same-species dry runs).
- **The memo's warning, accepted:** the set a dry run is expected to fail by construction (CK1: supercluster Fibroblast cannot reach 200 confident calls under any D2 option; on set a's self-map, NP3 class-balanced already fails broad Astro and Oligo). The registered path then applies: the criteria are revised in a PR the user approves, recorded before any new family is scored, and the dry run is re-run (CK2).

### 23.20 The family's NP5 depth profile: builder readings, and how the rulings of 2026-10-07 were implemented (recorded 2026-10-07, before any output of the family is read)

No output of the family has been read. This subsection changes no threshold, target or registered criterion. It records the readings the NP5 depth-profile builder takes (memo CHECK CK3), each an **adopted default, put to the user before any of the family's outputs are read**, and how the code applies the rulings of §23.19.

**The builder.** `scripts/annotation/build_np5_depth_profile.py` reads the family's `map_first` RESOLVE label tables (one per section, read only) and writes the asset `np5_depth__<family_id>` with its sidecar into the packaged `sim_inputs` directory, and a source manifest under the evidence root (each table's path, size and sha256; the sidecar names the tables by file name, section and sha256 only). The asset has its own role, `gate_p_profile`: one row per class and distinct total count with its cells, for the confident broad calls of each class and, apart, for the family's other table cells. Gate P alone reads it (`--depth-profile-asset`): PREP selects `profile` assets only, so the asset never enters a `build_hash` and the gate-P store's PREP stays that of the production bundle (`COMPUTE_NEXT.md` C18 item 4). The builder refuses a table that is not a human `map_first` RESOLVE output of a provisional panel, tables of another segmentation, panel, family or primary bundle, a section given twice, and a confident broad call without a floor class. *Added 2026-10-07 after the review of these changes* (none is a threshold or a criterion; each sends a case the readings do not cover back to the user): it also refuses tables that are not exactly the sections the command names (`--sections`, each by its sample id or pair id; reading 1 pools the four), a section whose dataset gate failed (it adds no confident call, but its cells would move the table median and NP3's report-only histogram) or records no gate level, and a table resolved for another region than `frontal_cortex` (D3 (a)). Gate P refuses the asset when the primary bundle its tables were resolved with (`run.primary_build_hash` in the sidecar) is not the one gate P's own PREP built, after PREP and before any leave-one-donor-out build or mapping, so the comparison of `COMPUTE_NEXT.md` C18 item 4 is made by the code. The asset is built, committed, frozen and recorded here only after the family's `map_first` re-run with the code of these rulings (§23.19; `COMPUTE_NEXT.md` C18).

**Readings** (CHECK CK3; a reading the memo, its CHECK or an adopted recommendation (A45 (a), B4 (a)) points to is taken, otherwise the strictest one named. For (ii) and (iv) neither reading is nested in the other, so none is the strictest by construction and the reading the adopted text names is taken. Each is an adopted default, put to the user before any of the family's outputs are read):
1. **The segmentation and the sections (CK3 (i)).** The scored segmentation, proseg_hybrid (D25, §23.4), with the four sections pooled, each table cell once (D8 (b), confirmed in §23.10, which CK3 (i) cites). Each section's counts and medians are reported in the sidecar and the manifest, and decide nothing. A table of another segmentation is refused.
2. **A class with 1–99 confident broad calls (CK3 (ii)).** It takes the overall median, as a class without calls: plan §8.3 v7.5's profile minimum of 100 cells, which the CHECK cites, which B4 (a) reuses for the class-depth prediction's fallback, and which `sim_inputs.DepthProfile` applies when it samples. The count is pooled over the sections. The asset keeps such a class's calls (they count in the overall median and are reported), and `sim_inputs.FamilyDepthProfile.class_depths` applies the minimum; an asset recording another minimum is refused. *Alternative:* the class's own median from its few calls. Neither reading is nested in the other.
3. **The finer levels (CK3 (iii)).** *Correction of fact to CK3 (iii):* gate P keys the decisions of every level, lineage to supercluster, by the E2 floor classes (`resolvability.whb_level_specs`: neurons by NT, glia by broad class, COP apart at supercluster), and NP5 takes a class's depths by that key. So an NT or supercluster class already takes the depths of the confident broad calls of its floor class (supercluster Astro those of broad Astro; NT Exc those of broad Exc), not the overall median. Only supercluster COP, which has no broad key (broad calls of COP cells are OPC), and any class without enough calls take the overall median (A45 (a), §23.17 item 7: "the profile's overall median gives every other class's"). *Alternative for COP:* its broad parent's (OPC) depths. The two are not nested.
4. **Which overall median (CK3 (iv)).** The median of the confident broad calls of every class, pooled over the sections: "the profile's overall median" of A45 (a) and §23.17 item 7, and D8's "overall median for classes without calls" in a sentence about confident broad calls. The label-free median of every table cell (D8 (b)'s depth) is reported beside it and decides nothing. Neither is nested in the other for NP5: a class that takes the overall median is judged at that one depth (its shares are 0 or 1, A7), and which of the two medians falls in an extrapolated bin depends on the decisions (pooled bins beyond D_P, usually the deep ones, or monotone-filled bins, which can lie between emitted bins). Confident calls lie deeper than the table cells, so where the extrapolated bins are the deep ones the adopted median is the stricter.
5. **NP3's report-only depth histogram.** With the family's asset, NP3's depth-histogram values (report-only, A3 (a): "only the label-free depth-histogram values are reported") use every table cell of the family, label-free, not its confident calls. (With a public per-class `profile` asset gate P still takes the profile's cells; none is used for this family.)

**How the rulings of §23.19 were implemented** (branch `feature/rca-m13-new-panel-onboarding`; none of this reaches the running set a dry run, which runs the frozen export of `2a91a36`):
- **C1 (b):** the family's referee uses the `class` comparator; the thresholds stay 0.75 / 0.70. *Corrected 2026-10-07 after the review of these changes:* the first implementation changed the default comparator for every human RESOLVE run, while §23.18 puts NR5 to the user "for this family" and the memo calls C1 "config values for the family's run". `real_qc.marker_referee_comparator` stays `node`, the C13 specification, and `real_qc.marker_referee_comparator_by_family` gives the family's id (`human_merscope_aa25d5a241d0`, D16 of the frozen D17 panel) `class`. RESOLVE reads the panel's family, so the family's `map_first` re-run takes `class` with nothing set in its config, and the seeded set a sections and any later family keep `node` (for set a, `not_evaluable`, C17). The outcome's settings record the comparator and the family it was ruled for (`comparator_family`).
- **C2:** `real_qc.registration_g1_effect` now defaults to `gate_failed` (§8.8's `failed` + `exclude_hard`), for every human RESOLVE run. D23 (b) made G1 warn-only in M13 until C17 showed no false failure, not for the family alone, so keeping warn-only elsewhere would extend that loosening beyond its approval (CK16). Registration G1 is not one of the seeded warn-only checks (§8.8, D20 (b)): a G1 fail on a seeded set a section now fails its dataset gate, and the C17 regression found no G1 fail rule on set a. This scope is the reading of D23 and C2 taken here; it is put to the user with the gate-P PR.
- The C17 regression script pins the settings it ran with (the `node` comparator for every family, G1 warn-only, NR7 at the lower edge), so a re-run reproduces C17.
- **C3 (b):** NR7 (`real_qc.gene_complexity_check`) now matches each native depth bin inside the grid by interpolating each test cell's simulated genes, linearly in log depth between the bin's edges, to the bin's native median total (`real_qc.gene_complexity_matching = interpolated`). Only test cells simulated at both edges are matched, and the 50-cell minimum counts them (C4 (a)). The open top bin keeps the lower edge.
- **B2 (b), as corrected by CK18:** gate P now reads NP2's weak and collapsed parents from PREP's bundle (reference data only) before any leave-one-donor-out build or mapping. A parent that `--gate-p-accepted-parents` does not name stops a family's run there (status `stopped`, exit status 3), so any parent other than those the user accepted comes back to the user before gate P runs. An accepted entry may name a parent by its lookup key, its node label or its node's name, so the user's "Bergmann glia" and "Upper rhombic lip" match their keys; an entry that matches two parents is refused, and one that matches none is reported. A dry run does not stop (NP2 is not part of its rule), and `--gate-p-run-with-unaccepted-parents` runs anyway, with NP2 pending. Whether the two parents are plausible in frontal cortex plays no part (CK18).
- **A48 (b):** no code change. If the pool-size stop fires, the run is re-run in place with `--gate-p-other-region shared`, for the v6 and v7 dry runs and for the family (`COMPUTE_NEXT.md`).
- **D3 (b):** the plan's M13 exit is amended (plan §12 M13 and M13b).
- Every other ruling keeps what the code did at `857d75c`.

### 23.21 The set a dry run, the criteria revision and the user's approval (recorded 2026-10-07, before the revised code is written and before the dry run is re-run; no output of the family has been read)

**The dry run as registered.** Set a, code `2a91a36` (an exported tree), other-region draw `shared` (A48 (b)), expected depth 33:
- **Version 6:** scored, `dry_run.passes` False. Every broad C_P class fails: Astro NP3; Exc and Inh NP6; Immune and OPC NP5; Oligo NP4. At supercluster, Fibroblast and Vascular are `not_evaluable` (CK1).
- **Version 7:** the first attempt crashed on a full storage RAID (another user's copy). The re-run (`v7_r2`, same code, a fresh store) scored `dry_run.passes` False: every C_P class fails at every level, mostly NP5 and NP6. `measured_seconds` 4,618.
- **NP9 identity** failed in both runs on a code bug, not a criterion. The marker lookup's MapMyCells run record (scratch paths, worker count, times) was compared. Fixed in `1a8fe2e`.

**Causes** (memo `$A/m13/DRYRUN_REVISION_MEMO.md`, sections 2 and 6). A read-only re-scorer reproduced both runs cell for cell from the saved replicate frames (`$A/m13/dryrun/revision_whatif_v6/`, `revision_whatif_v7_nc/`):
1. NP3's scored per-call-set weightings (A1 (a), §23.9 item 2) estimate no precision: a few stray wrong calls take the trim cap. Broad Astro at 120: 0.875, against 0.98–0.99 on the test-cell readings. NP6 reuses them: 215 failing rows, against 0 unweighted at broad and supercluster.
2. NP5's t* spread ≤ 0.05 measures sample size. Two mapping seeds on the same cells already differ by > 0.05 in 22% of glial sets and 1.4% of Exc/Inh sets.
3. NP4–NP7 fail a class at sets below its NP3 floor.
4. NP5's agreement compares a bin when either side has ≥ 50 test cells. At bin 250 the base held 4–21 cells, so its status there was a pooled extrapolation.
5. Version 7: NP5 compares each member's own decisions (the C8 reading), while NP3/4/6/7 score the frozen ensemble emission.

**The revision approved by the user** (chat, 2026-10-07, answer to V1: "Approve all six (Recommended)"). Every item is a LOOSENING and POST-HOC: its effect on set a was computed before it was proposed.
- **R1:** NP3's two weightings are read on the test cells of the set's scope (per called cell π_t / N_t, trimmed as before). NP6 uses the same weightings. This replaces A1 (a) and §23.9 item 2's per-call-set reading.
- **R2:** `validated_min_depth` is the shallowest bin from which every tested set, up to and including the "≥ D_P" set, passes NP3 and NP4–NP7.
  - A failure in the ≥ D_P group fails the class, as does a failure of a class-level part (NP4's seed criterion, NP5's extrapolated share, NP7's excluded share).
  - A failure at a shallower set raises the floor.
- **R3 (c):** NP5's t* part is a consequence check: each replicate's own t*, applied to the set's pooled base calls, gives point precision ≥ target_L − 0.02. In version 7 the check is applied to the ensemble's pooled t*, refitted per replicate.
- **R5:** in the dry run, a class H18 expects whose record is `not_evaluable` is reported, not failed. This replaces CK1 (a).
- **R6:** NP5's agreement compares a bin only when the base and the replicate each have ≥ 50 test cells there.
- **R7 (version 7):** NP5's agreement compares the ensemble emission re-derived per replicate (`np5_rederive_ensemble`) with the frozen ensemble decisions, instead of each member's own. R6 and R2 apply to it. This replaces the C8 reading.
- **Unchanged:** targets, target⁺, Wilson on Kish n, coverage 0.30, n_min, NP4's floor and range rules, NP6's stresses and drop test, NP7.

**Predicted effect** (re-scorer, from the saved frames; the re-run on the revised code must reproduce it):
- both dry runs pass, with `validated_max_level` supercluster;
- version 6 broad floors: Astro 30, Exc 10, Inh 10, Immune 10, OPC 30, Oligo 60;
- version 7 broad floors: Astro 30, Exc 10, Inh 10, Immune 15, OPC 30, Oligo 120.

**Negative control** (V2, user: "Enough; state fragility (Recommended)"):
- **Panel:** set a minus 25 Astrocyte- and OPC-specific genes. The list and its expected outcome were recorded in the memo before it ran: `$A/m13/dryrun/negative_control/`, sha256 434720fa… and 35858645….
- **Version:** it ran as version 7 (a new family, `version: auto`), not version 6 as the memo first said.
- **Result under the six revisions, against set a version 7:**
  - broad OPC fails, decided by several members (truth-Oligo cells called OPC at 120);
  - broad and supercluster Astro fail through member @12 only (NP6 spill stress), so these results are fragile;
  - supercluster OPC is validated as on set a.
- **Reading:** the expectation holds for 3 of 4 cells and both broad cells, so the recorded rule ("if neither moves") calls for no tightening. The gate is not blind, but this control does not show it is sharp. The revision PR states this.

**Other answers of 2026-10-07 (same exchange):**
- **V4, "Every human run (Recommended)":** C2 (registration G1 fails the dataset gate) applies to every human RESOLVE run, as implemented. No set a section fails G1.
- **V5, "Confirm all four (Recommended)":** the NP5 profile builder's readings (i)–(iv) of §23.20 are adopted.
- **B1:** the family's frozen panel is `panel_hash` `aa25d5a241d0b273ed855a83c8a0310a86e3242cb6f43345f1736ace846cdd62`.

**Next (CK2):**
1. The revision is implemented with tests.
2. The dry run is re-run, v6 then v7 (fresh v7 store), on an export of that commit.
3. Both must pass and reproduce the predicted records.
4. Only then is any new family scored.

### 23.22 R3 (c) in version 7: the ensemble's t* is read at every bin of a tested set (recorded 2026-10-07, after the review of the revised code; before the dry run is re-run)

This subsection changes no threshold, target or criterion. It records how the code reads one phrase of §23.21 R3 (c) that the text does not settle.

**The phrase.** In version 7, R3 (c) is "applied to the ensemble's pooled t*, refitted per replicate": the threshold of the replicate's re-derived ensemble decisions (`np5_rederive_ensemble`) "at each tested set". A "≥ D_P" set spans several bins, and the replicate's ensemble can decide them on different sets. For example, D_P may be decided on its own and a deeper bin on the replicate's deep pool, each with its own threshold.

**The first implementation** (`8e4479e`) read one threshold per (set, replicate): the decision at the set's shallowest bin, applied to every call of the set. That never applies the replicate's thresholds at the deeper bins. A review counter-example: the replicate applies 0.85 at 120 and its pool's 0.55 at 250. Its emission of the "≥ 120" set's base calls has precision 0.85, below 0.88, but the shallowest-bin reading gives 1.0 and passes the set.

**The reading taken** (`np5_ensemble_set_thresholds`, `np5_tstar_consequence`): each base call of the set is kept at its own bin's threshold in the replicate, as the replicate's emission applies them.
- A bin of the set whose decision is fitted (`n_fit` ≥ `min_cells_per_bin`) but has no threshold fails the set (`missing`: the replicate would emit nothing there), as a fitted member without a t* does.
- An unfitted bin emits none of its calls.
- A replicate with no fitted bin in the set is left out (`unfitted`).
- A single-bin set reads as before. The member form (version 6, and the report-only cross-check) is unchanged: one t* per (set, replicate), applied to every call of the set.

**Why this reading.** It is what the replicate's ensemble would emit, so it asks R3 (c)'s question ("does the threshold's wobble change what is emitted"): every call is held to the threshold the replicate applies to it. The readings are not nested. Where a deeper bin's threshold is lower than D_P's, the reading taken keeps more of that bin's calls than the shallowest-bin reading, so it is stricter there; where it is higher, it keeps fewer, so it is looser there. The alternative of the review, the worst single bin's threshold applied to every call of the set, holds calls to thresholds the replicate never applies to them.

**What was known when it was recorded:**
- On set a version 7 (`v7_r2`), a review probe (`$A/m13/tmp/revision_impl/review_adv_stats/`, read-only on the saved frames) computed three readings:
  - the shallowest-bin reading;
  - each call at its own bin's threshold (the reading taken);
  - the worst single-bin threshold.
  None fails a set; the smallest margin is +0.116 to +0.117. 240 of the 288 (member, pooled set, replicate) rows have more than one threshold across their bins, and none has a fitted bin without a threshold.
- The negative control's effect had not been computed under this reading.

**Status.** This is the reading the code scores. It is put to the user with the revision PR, beside §23.21.

### 23.23 The set a dry run re-run on the revised code passes (recorded 2026-10-07, before any output of the family is read)

Code `a3c0707` (§23.21 and §23.22 implemented), exported to `$A/m13/dryrun/code/head`; other-region draw `shared`; expected depth 33. The registered dry run's code is kept at `code/head_2a91a36`.

- **Version 6 (`v6_rev1`):**
  - `dry_run.passes` True; `validated_max_level` supercluster.
  - Supercluster Fibroblast and Vascular are reported as not evaluable (R5).
  - NP9 identity passed (the `1a8fe2e` fix).
  - All 27 records equal the re-scorer's prediction (§23.21).
- **Version 7 (`v7_rev1`, fresh store `gate_p_store_v7_rev1`):**
  - `dry_run.passes` True; `validated_max_level` supercluster.
  - All 27 records equal the prediction.
  - Broad floors: Astro 30, Exc 10, Inh 10, Immune 15, OPC 30, Oligo 120.
- **NP9's time reference for the family (D10 (a)):** version 7's `measured_seconds` 4,259.4 for 6,744,671 simulated cells.
  - At its start the host's load was 2.85 / 4.24 / 38.15 (1 / 5 / 15 min), falling.
  - Start was held until the 15-minute load fell below 40, after another user's job had held the load at about 350.
  - The family's run is to be started at a similar load, and both loads are reported in its gate-P PR.
- **Still open in the family verdict, as expected for a dry run:** NP2 pending (set a's Vascular parent, never accepted) and NP9 not evaluable (time: no reference for set a itself). Neither is part of the dry-run rule (D28).

CK2 is met. Gate P may now run on the new-panel family (C18), after its PREP and `map_first` re-run on this code or later, and the NP5 profile asset.

### 23.24 The family's re-run and its NP5 depth profile (recorded 2026-10-08, before gate P runs on the family)

**The re-run.** The family's PREP and `map_first` re-ran on code `a070e05` (export `$A/m13/chain/head_a070e05`), replacing the `_m13` tables of 2026-10-06 (user's choice, 2026-10-07: "Start now, replace _m13", after a full backup). Run outputs are under the family's `acceptance/2026-10-07/runs/`.
- **One task failed:** `CLUSTERING_SQUIDPY_FINALIZE` (HM2, proseg_mask) read a table while another task was replacing it.
  - **Cause:** the store write lock is keyed on the unresolved path, and each Nextflow task reaches the store through its own work-dir symlink, so the writers were not serialized. The fix goes to `main` separately.
  - **Recovery:** HM2 re-ran alone (run label `npf3b`, store writers serialized with `maxForks = 1`).
  - **Checks:** no failed task; no table file outside the `_m13` tables changed; every `_m13` table was rewritten. HM2's four `_m13` tables carry `ct_*` columns identical to the first run's, and its proseg_hybrid label table is identical in content (only the parquet provenance metadata differs).
- **The family's primary bundle:** `whb_frontal_supc_clus` `build_hash` `0e32a8f6ce934cdb0da2af581e03a225e92a8131b90c34f1802d08485e075113`. Gate P checks its own PREP bundle against this before any build (`e95d66e`).

**The NP5 depth profile.**
- **Asset:** `np5_depth__human_merscope_aa25d5a241d0` (role `gate_p_profile`), sha256 `7902680605779eb0511b0bf47ad313c7a853b9cb3b1d0edc10244934d3fc4e44`, committed in `20fb4c7`.
- **How it was built:** `build_np5_depth_profile.py`, with the readings (i)–(iv) of §23.20 that the user confirmed on 2026-10-07 (V5), from the four sections' proseg_hybrid label tables of the run above, pooled.
  - 679,343 table cells; 390,686 confident broad calls.
  - Classes with their own depths: Astro, Exc, Immune, Inh, OPC, Oligo, Vascular.
  - Fibroblast takes the overall median.
  - Overall median: 328 counts (confident broad calls). The label-free table median, reported only, is 236.
- **Source manifest:** `$A/m13/npf_run/np5_profile/source_manifest.json`.
- **The first read of the family's outputs:** building this asset. Nothing else of the family's outputs has been read.

### 23.25 Gate P on the new-panel family: result (recorded 2026-10-08, when the run ended)

**The run.** Code `2c25541` (export `$A/m13/npf_run/code/head`), at host load 5.3 / 12.1 / 11.1 at its start (the v7 dry run's was 2.9 / 4.2 / 38.2). Run outputs: `$A/m13/npf_run/gate_p_run/`.
- **First attempt:** under D2 (d), it stopped before any mapping: 6 (donor, class) pools could not meet the per-class top-up rule.
- **Re-run in place** with `--gate-p-other-region shared`, as A48 (b) rules (§23.19). It scored in 1 h 42 min.

**Verdict: the family does not pass gate P.** `validated_max_level` none, because lineage is not complete. The family stays `provisional` (§14 Gate-P rule). The real-data onboarding (§23.7) is separate.
- **Family checks:**
  - NP1 passed.
  - NP2 passed: no weak or collapsed parent; the two names the user accepted matched none.
  - NP8 not applicable.
  - NP9 passed: `measured_seconds` 5,003 against the reference 4,929 × 1.5 (set a's v7 dry run, scaled to 7,805,264 simulated cells), RSS within the reserve, identity identical.
- **Lineage: every class fails NP7's 1% part** in 7 of the 8 members.
  - Confident calls to excluded nodes are 1.12–2.92% of the level's confident calls; member @9 is at 1.00% and passes.
  - Almost all of these calls go to the WHB sink Miscellaneous (CS202210140_463), counted confident by the larger of the call's own lineage bp and its supercluster bp (§23.13 item 1, ruled A8 (a)).
  - At broad, NT and supercluster the same share is 0.09–0.69%, which passes.
- **Broad:** 5 of 6 C_P classes validated. Broad Oligo fails NP3 in member @6 only: class-balanced precision 0.948 at the ≥ D_P set (bin 250), against target⁺ 0.95.
- **Supercluster:** all 6 C_P classes validated (Exc only from depth 250).
- **NT:** both classes validated.

**No revision.** These criteria passed the set a dry run, so the registered rule (§9 "Dry run"; CK2) gives no ground to revise them for this family. Changing NP7 or NP3 now would be a post-hoc loosening that validates more classes (§1 rule 2). Any later revision applies to the next families only, after a dry run.

### 23.26 The family's real-data onboarding: results (recorded 2026-10-08, after gate P)

**What was read and run.** The family's run of 2026-10-07 on code `a070e05` (§23.24), read only from `<family results>/acceptance/2026-10-07/runs/` (HM2 from its NPF3b re-run, whose `_m13` tables are in the store; its NPF3 and NPF3b label tables are identical in content on all four segmentations). Re-runs used the run's own code export, task scripts and staged inputs, CPU only, into scratch under `$A/m13/npf_acceptance/`; no results store or published run output was written. Evidence: `$A/m13/npf_acceptance/ACCEPTANCE_REPORT.md`, its `tables/` and `scripts/`. No threshold, reading or criterion was changed.
- **NR1:** RESOLVE re-run on each section x segmentation's published MAP output, as published (QC on) and with `real_qc.enabled` false.
- **NR2 (a):** the QC-on re-runs compared with the published outputs; MAP re-run on HM1 proseg_hybrid at the published seed (0, 6 workers) and RESOLVE on it.
- **NR2 (b):** MAP with `rng_seed` 1 on all four sections x proseg_hybrid, then RESOLVE; scored with M8's `seed_change` (gate H's reading, §20 D12 (a)).

**Rows** (proseg_hybrid, the scored segmentation; values in the order HM1 / HM2 / HM3 / HM4):

| Row | Value | Threshold | Effect | Outcome |
|---|---|---|---|---|
| P1 | One `panel_hash` (`aa25d5a241d0…`) in every label table and manifest. No `Blank-*` feature in any store table, prepared panel or MapMyCells query (496 query genes on both references). The 65 blanks are dropped at ingestion, upstream of the registry, which therefore removed 0 | §23.5 | `UNSCORED` if not | holds |
| P2 | No `validated_panels.csv` row; trust `provisional` (`family_not_validated`) | §23.5 | `UNSCORED` if not | holds |
| P3 | RESOLVE consumed the version-7 bundle; MAP and RESOLVE name `0e32a8f6…` (WHB) and `23a89f5f…` (SEA-AD) | §23.5 | `UNSCORED` if not | holds |
| P4 | All ten §8.8 checks carry an outcome token on every section x segmentation | §23.5 | `UNSCORED` if not | holds |
| NR1 | No violation on 16 of 16 section x segmentation (`real_qc.downgrade_only_violations` of the run's code, and every difference listed). Trust state, gate level (`full`), emission plan (resolvability record), thresholds and floors, and every label column are identical. The only differences are the QC's gate warnings, each named by its check (`real_qc_coverage_vs_simulation` x22 / 21 / 23 / 24, `real_qc_flag_rates` x1 each, `real_qc_gene_complexity` x1 on HM1 and HM3), and the QC provenance records | Identical except labels set to `not_resolvable` and a gate level lowered by a named check | The M13 exit fails | **pass** |
| NR2 (a) | The QC-on re-runs give label parquets byte-identical (sha256) to the published ones, with identical summaries and manifests, 16 of 16. The MAP re-run's three parquets are identical in content (its manifest differs only in `created_at`, wall times and RSS), and the label table RESOLVE gives on it is identical in content | Identical label-table content | The M13 exit fails | **pass** |
| NR2 (b) | Switched between two confident names: 0 of 131,388 / 63,860 / 102,118 / 93,320 seed-0 confident broad labels (0%). Threshold crossings, reported: 5.25% / 6.05% / 5.37% / 4.50% | Switched ≤ 1% | A warning | pass |
| NR3 | PANEL + MAP + RESOLVE wall over the four segmentations (H14's sum): 0.30 / 0.20 / 0.24 / 0.22 h (every non-PREP task of the section: 2.04 / 1.14 / 1.43 / 1.38 h). MAP peak RSS 11.3 / 10.2 / 10.6 / 10.4 GB. No GPU process; GPUs hidden in all 146 task scripts. PREP, reported: WHB 385 s, SEA-AD 487 s (NP9's PREP part 758 s) against NP9's limit of 1.5 x 4,929 s for PREP and the replicates together (gate P measured 5,003 s) | ≤ 2.5 h; ≤ 16 GB; no GPU | A warning | pass |
| NR4 | Gate `full` on all four. A 0.928 / 0.884 / 0.979 / 0.969. Confident broad coverage of table cells 0.504 / 0.537 / 0.671 / 0.634, of segmented objects 0.465 / 0.504 / 0.665 / 0.625 | `broad_only` A < 0.30; `failed` < 0.25; warning < 0.15 | As defined; `failed` is a concern | pass |
| NR5 | `class` comparator (C1 (b)), 7 marker groups: 0.968 / 0.982 / 0.957 / 0.975 (other segmentations 0.951–0.977) | Warning < 0.75; `broad_only` < 0.70 | Gate cap | pass |
| NR6 | Uninformative strata, literal reading: 6/7, 4/7, 7/7, 5/7. Option (a), reported: 5/21, 4/21, 6/21, 5/21, which would not warn | Warning > 0.5 | A warning | **warning** on all four |
| NR7 | Largest gap 0.55 / 0.41 / 0.78 / 0.35, always at the open top bin (250, matched at its lower edge). The other bins lie between −0.25 and +0.04 | Warning > 0.45 | A warning | **warning** on HM1 and HM3 |
| NR8 | (Level, class) with ≥ 200 cells below the prediction − 0.10: 22 of 33, 21 of 29, 23 of 33, 24 of 33. At broad every judged class falls short except Oligo on all four, Exc on HM3 (−0.09) and Fibroblast on HM1 (−0.06; on HM2 it has 150 cells and is not judged) (Astro real 0.36–0.61 against 0.87–0.94 predicted; Oligo 0.55–0.75 against 0.54–0.72). The largest shortfall is 0.54 at broad (HM1 Astro) and 0.60 at any level (HM4 supercluster Inh). `flag_nonneuronal_high_depth` is false for every cell (§22.9 (c)); the OPC and Oligo high-depth trend fires, report-only | Warning when real < simulated − 0.10 | A warning | **warning** on all four |
| NR9 | `not_evaluable` in every RESOLVE ("no registration check given": no M0a check exists for these sections). G1 is a per-section check (§7.6), so it is applicable but was not run. Computed read-only for this record: density ratio 1.75 / 3.05 / 2.18 / 3.16, shift 0 µm, no fail rule on any section x segmentation; HM1 would warn (< 2.0) | Fail < 1.5 or shift > 5 µm; warning < 2.0 | `failed` + `exclude_hard` (C2) | not evaluated |
| NR10 | Confident COP supercluster 0 on all four. Confident broad OPC 0.98% / 1.35% / 1.19% / 0.51%. COP-derived OPC share, reported: 4.9% / 3.7% / 2.6% / 6.6% | ≤ 2%; ≤ 10% | Concern on ≥ 2 sections | pass |
| NR11 | `flag_implausible` 0.50% / 0.34% / 0.88% / 0.28%, mostly the WHB sink Miscellaneous. Reported segmentation reseg: HM3 1.28% | ≤ 1% | Concern on ≥ 2 sections | pass |
| NR12 | 0.865 / 0.892 / 0.913 / 0.900 (recomputed from the MAP outputs, equal to the report's) | ≥ 0.80 | Concern on ≥ 2 sections | pass |
| NR13 | Gate `full` on all four. Confident broad coverage of table cells 0.504 / 0.537 / 0.671 / 0.634. **No predicted levels were recorded**: no label-free depth read was written before the family's MAP or RESOLVE outputs were read (§23.24: the NP5 asset was the first read) | Report-only | None | report (prediction missing) |
| NR14 | Share of confident broad calls: Oligodendrocytes 0.55 / 0.49 / 0.30 / 0.64, Neurons 0.16 / 0.17 / 0.33 / 0.10, Vascular 0.09–0.17, Astrocytes 0.05–0.14, Microglia 0.03–0.09, OPC 0.008–0.025, Fibroblasts ~0. reseg vs proseg_hybrid on shared cell ids: 74% / 86% / 95% / 95% of the proseg_hybrid table cells have a reseg table cell; among cells confident in both, the broad name agrees in 99.74–99.87% (supercluster 99.87–99.91%) | Report-only | None | report |
| NR15 | Depth check `not_run`: no pia / WM boundaries exist. Protein IF: the channel names name GFAP, SNAP25, MBP and CLU, none of which is on the RNA panel, so a per-gene concordance cannot be computed. Reported instead (a reading taken after the run): the cellpose-mask intensity of each channel in the RNA-confident broad class it marks, AUROC against the other confident cells: GFAP 0.62–0.95, SNAP25 0.69–0.94, MBP 0.85–0.91, CLU 0.52–0.80. Held-out-gene enrichment `not_run`: no marker list was fixed before the run | Report-only | None | report |

**A probe of NR1's lowering path, report-only.** The published QC lowered nothing on the family, so the NR1 comparison above never exercises a lowering. A probe re-run raised the referee's thresholds to 0.99 / 0.995 (not registered values; never a published run). The gate then fell to `broad_only` by `real_qc_marker_consistency` on all four sections. Only the gated levels' statuses (to `not_attempted_gate`) and the columns derived from them changed; trust, emission and thresholds did not. The same checker found no violation.

**Verdict (§23.7): Onboarding accepted.** P1–P4 hold, and NR1 and NR2 (a) pass on all four sections x proseg_hybrid (and on reseg, proseg_mask and original_seg). The warnings (NR6, NR7, NR8) are explained in the PR.

**Family-level concerns (§23.7): none fires.**
- NR4 is `failed` on no section.
- NR5 is `broad_only` on no section (lowest 0.957).
- NR10, NR11 and NR12 fail on no section.

The family stays `provisional`: gate P did not pass (§23.25), and real data never promote (§23.3 rule 1). Points for the user beyond the triggers:
- NR8's shortfall is large and family-wide: simulation over-predicts real coverage in 21–24 (level, class) rows per section, by up to 0.54 at broad and 0.60 at supercluster. At broad only Oligo (all four), Exc on HM3 and Fibroblast on HM1 are within 0.10.
- NR9 was never evaluated in the run.
- NR13's prediction was not made before the outputs were read.

### 23.27 M13 closed; follow-ups of the new-panel family (recorded 2026-10-13)

**Status.** Recorded after §23.26, with the user's answers of 2026-10-13 to the numbered list of next steps put to the user that day. It changes no threshold, target, criterion or verdict, and no published output. The analyses below read the family's and the human sections' published outputs and wrote only under `$A/m13/followups_20261013/`; each states what it read. Where an analysis proposes a change, the proposal is labelled and its status stated.

**1. The user's decisions of 2026-10-13** (verbatim):
- Closing M13: "1 - yes this is now complete, close M13".
- The new-panel family's follow-ups: "3 - proceed as you see fit, but on the depth test sides - exclude these. The tissue annotations wont come for some time and you should hold up completing this based on that."
- The two cortical-depth drawings of HD1_IF: "for 2 the later drawing is correct".

**2. M13 is closed on the human path** (the exit of §23.19, D3 (b)).
- The exit items are met: the set a dry run passed on the revised code in both versions (§23.23); the family's NP report and its documented gate-P failure (§23.25); onboarding accepted (§23.26); the human rows of the real-data QC regression (C17, §23.18). The code and records are on the integration branch (PR #42, 2026-10-08).
- The family stays `provisional`, `validated_max_level` none, with no `validated_panels.csv` row (§23.25). Real data never promote it (§23.3 rule 1).
- Not part of M13, by the user's answer: the human Xenium Prime 5K and Xenium Human Brain (266) gate-P runs (D4 (a), "after this family"; §23.2 listed them as unanswered). They are a later step with no milestone yet, and the user starts each run (D25).
- The mouse path (C20–C22) is M13b, after M6b (§23.19).

**3. §23.22's per-bin reading of R3 (c) is adopted** ("proceed as you see fit", item 3).
- It is the reading the code scored: the set a dry run (§23.23) and the family's gate P (§23.25) both ran under it, so no verdict changes.
- It holds each call of a tested set to the threshold the replicate applies at that call's own bin. §23.22 states that it is not nested with the first implementation's shallowest-bin reading: it is the stricter of the two where a deeper bin's threshold is lower than the shallowest bin's, and the looser where it is higher. On set a version 7 no set fails under either (smallest margin +0.116).
- No threshold, target or criterion changes, and §23.22 recorded the reading before the dry run was re-run and before any output of the family was read, so it is not post-hoc; §1 rule 2 is not engaged.

**4. NR9 evaluated: registration G1 fails on none of the 16 section × segmentation rows.**
- **Method.** A `git archive` export of the integration branch at `1164b2f` (`$A/m13/followups_20261013/code/head_1164b2f`). The QC stage's own `compute_segmentation_registration_qc` ran with `merscope_cell_boundaries` as the reference, on the published stores, reading only the shapes and points parquets. A full-store `sd.read_zarr` run on HM1 proseg_hybrid matches the written record exactly, and all 16 records match the read-only values behind §23.26's NR9 row. The stores and RESOLVE were not written to or re-run.
- **proseg_hybrid** (the scored segmentation):

| Section | M0a status | Density ratio | Shift | G1 |
|---|---|---|---|---|
| HM1 | pass | 1.749 | 0.0 µm (8 of 8 windows) | warning (< 2.0) |
| HM2 | pass | 3.055 | 0.0 µm | pass |
| HM3 | pass | 2.185 | 0.0 µm | pass |
| HM4 | pass | 3.157 | 0.0 µm | pass |

- **The other segmentations** (HM1 / HM2 / HM3 / HM4): reseg 1.915 / 2.864 / 2.299 / 3.011 and proseg_mask 1.724 / 3.004 / 2.143 / 3.087, shift 0.0 µm on both; original_seg 2.781 / 2.882 / 2.179 / 3.166, shift not measured (its reference is its own shapes). Only HM1 on reseg, proseg_mask and proseg_hybrid is below 2.0, and its lowest ratio, 1.724, is above the fail line of 1.5.
- **NR9: pass.** The fail rule (ratio < 1.5 or shift > 5 µm; effect `gate_failed`, §23.19 C2) fires on none, so C2 has nothing to act on.
- **What G1 means for the gate levels.** The gate stays `full` on all 16, with no `exclude_hard` and no label change. HM1 on the three segmentations above would carry a G1 warning flag on its dataset gate; the other 13 rows would be clean. The gate-P and onboarding verdicts (§23.25, §23.26) stand, and §23.26's NR9 row stays as recorded (`not_evaluable` in the run, outcome "not evaluated").
- **Published outputs.** They still hold no G1 outcome: no registration check file is in the family's results (§23.26), and RESOLVE was not re-run. The 16 records are evidence in the form RESOLVE's lookup reads, `nr9/<segmentation>/p<id>_merscope_registration_qc.json`, with `nr9_g1_summary.{csv,json}`, `compute_nr9.py` and `qc_stage_crosscheck.py` beside them, under `$A/m13/followups_20261013/nr9/`.

**5. NP7's lineage sink rule: a proposal for future families, not adopted.** It reads the family's gate-P outputs and set a's version 6 and version 7 outputs (`v7_rev1`, 8 members; `v6_rev1`); it reproduces §23.25's NP7 rows exactly for all 8 members and re-scores nothing (`$A/m13/followups_20261013/np7_sink/`).
- **Why lineage fails and the other levels do not.** The same 1,727 cells (member @0) reach a sink at every level. Under §23.13 item 1 a sink call is confident on the larger of its own bp and its supercluster bp. At lineage the cell keeps its own lineage call, Neurons, with median bp 0.90 against 0.43 on its supercluster bp, so 1,407 of the 1,727 (81%) are confident there, against 83 at broad, 81 at NT and 114 at supercluster (5–7%).
- **Who they are.** Across the 8 members, 93% of the lineage-confident sink calls (11,701 of 12,637) are confident on that own bp alone. By truth class: Exc 91.9%, Inh 7.4%, non-Neuron truth 86 calls (0.7%); Miscellaneous takes 97%. So 99.3% are neurons whose lineage label, Neurons, is correct.
- **The family differs from set a.** Exc to a sink is 2.9–3.2% of Exc rows at every depth; on set a it falls from 1.3% to 0.11%. Three excitatory superclusters supply 89% of the family's calls.
- **Candidate readings** (lineage share over the 8 members; set a's, as scored: 0.00–0.016%):

| Reading | Lineage share | Label |
|---|---|---|
| As scored (§23.13 item 1) | 1.00–2.92%, 7 of 8 fail | reference |
| Count every call, whatever its bp (item 1's stricter reading) | 1.21–3.39%; set a v7 NT and supercluster fail in 3 of 8 members | tightening |
| Judge on supercluster bp only | 0.06–0.29%; counts 0 of the 86 non-Neuron calls | loosening, post-hoc |
| Count only the calls whose truth is not Neurons | 0.003–0.024%; counts all 86 | loosening, post-hoc |
| Leave lineage out of NP7's 1% part | not tested | loosening, post-hoc |

  Broad, NT and supercluster stay at 0.08–0.69% as scored.
- **Recommendation:** count only truth-not-Neurons at lineage, the one reading that still sees glia sent to a neuron sink. The single-wrong-node view with the excluded calls (§23.13 item 4) needs the same change: member @10 fails Exc sets 10–120 on a "Neurons" node that is the correct lineage.
- **Status.** Not adopted. It is a loosening (§1 rule 2) and post-hoc for this family, so it applies to future families only, after a new set a dry run (§9, CK2) and the user's written approval. §23.25 stands: the family's lineage failure is not revised.

**6. NR8's over-prediction: investigated; proposal of no criterion change.** All numbers are proseg_hybrid at broad, classes with at least 200 cells, unless stated (`$A/m13/followups_20261013/tables/`).
- **Depth does not explain it.** The gap is in every bin. 70% of its weight is in the open bin of 250 counts or more, where real coverage is 0.72 against 0.93 simulated; real cells there have median counts of 421–553, and the simulation is matched at 250, so that bin is conservative. HM1 Astro lineage confidence is 0.15 at 60 counts against a simulated 0.81.
- **By class.** Oligo is within ±0.03. Astro −0.45, Inh −0.35, Vascular −0.32, Immune −0.32, Exc −0.18.
- **Reweighting does not explain it.** An unweighted predictor warns more: 31 of 32 section × class rows, against 25 of 32.
- **Gene complexity (NR7) does not explain it.** Native cells have 4–25% fewer genes than simulated at 10–120 counts and match at 120, so it cannot account for the gap at 120 counts and above.
- **Segmentation explains most of it.** At 250–499 counts, non-Oligo lineage coverage falls from 0.84–0.96 in the smallest-area quintile to 0.25–0.51 in the largest. The median gap is −0.25 on proseg_hybrid, −0.26 on original_seg, −0.22 on proseg_mask and −0.13 on reseg. The same cells in reseg have 30–60% fewer counts and a higher confident share (+0.04 to +0.25 for non-Oligo), so the extra ProSeg counts behave like contamination.
- **Not specific to the new panel.** NR8 was `not_applicable` on set a; the equivalent on set a (version-6 bundle, validated regime) has 40 of 64 rows below −0.10 on proseg_hybrid (median −0.18) and 15 of 62 on reseg (median +0.06), with the same class pattern.
- **Proposed follow-up (no criterion changes; none adopted).**
  1. Diagnostic, about 0.5 day: add an area-quintile NR8 table and a paired hybrid-versus-reseg table to the acceptance script, and score NR8 on reseg too for future families.
  2. Optional report-only QC note, about 1 day: shown when the largest-minus-smallest area-quintile gap is 0.15 or more (here 0.26). RESOLVE would need the mask area.
  3. Mixed-cell stress recipe, held: real mixing is roughly 4 times the simulated. It would apply to later families only, after a set a dry run (§9); about 2–3 days and 3 h of CPU per family.
- **Reference-store finding.** The family's WHB bundle `0e32a8f6` in the SSD1 reference store (§2) has six files whose sha256 no longer matches its `bundle.json`; the `.h5` file and three parquets cannot be read. An intact copy is at `$A/m13/gate_p_store/whb_frontal_supc_clus/0e32a8f6…` (`bundle_integrity_check.txt`). A re-run from the SSD1 store fails until it is restored; nothing was written to the store.

**7. Depth checks.** By the user's answer on item 3 (quoted in 1 above), the family's depth checks are not run and not planned. NR15's depth check stays `not_run` until the pia / WM boundaries exist, and nothing in §23.27 waits on them. The text of NR15 and D26 (§23.2, §23.6) is otherwise unchanged.

**8. HD1_IF: the August drawing is correct.**
- **Which drawing.** The user's answer ("for 2 the later drawing is correct") makes the August drawing the correct one: `$MERXEN_RESULTS_ROOT/HD1_IF/merscope/annotations/cortical_depth_annotations.geojson` (sha256 `dd620957…`). The IF batch of 2026-10-02 (task `work_if/d0/4518cd…`) used the July drawing (`…/SSD2/MerXen/merscope/HD1_IF/cortical_depth/HD1_IF_cortical_depth_annotations.geojson`), so its published depth results are July-based.
- **Method.** Code export `head_57c8488` (the batch's), the batch task's config with only `annotation_path` changed, standalone `merxen compute-cortical-depth` with `write_spatialdata_table=false`, 6 workers instead of 8, all 4 segmentations; then the annotation-report checks per segmentation. A control run on the July drawing reproduces the published outputs exactly (per-cell depth max difference 0.0, identical ribbon membership, all 141 report metrics unchanged). Nothing under the results store or `acceptance/` was written.
- **The drawings differ** by a median of about 31–59 µm for the pia, WM and side lines (p95 up to 188 µm; side line max 735 µm). Piece_1 ribbon area is 53.96 mm² in August against 54.54 mm² in July.
- **Depth changes** (proseg_hybrid; the other three move the same way):

| | July | August |
|---|---|---|
| Streamlines (flagged) | 611 (17) | 609 (18) |
| Cells inside the ribbon | 149,224 (0.7527) | 148,008 (0.7466) |
| Upper IT median depth | 0.242 | 0.227 |
| Deep IT median depth | 0.645 | 0.648 |
| Deep NP/CT/6b median depth | 0.822 | 0.823 |
| Oligo WM - GM | 0.629 | 0.627 |

  Per-cell equivolumetric depth moves a median of 0.021 (Spearman 0.995), a median of 0.0096 deeper in July than in August.
- **Check verdicts.** The H12 ordering (Upper IT < Deep IT < Deep NP/CT/6b) passes on all 4 segmentations with both the tangential-block and the square-tile CIs, and Oligodendrocytes WM > GM passes on all 4, as they did with July. The depth QC warning `abnormal_local_thickness_variation` persists on all 4. The structural checks (8 violins, `map_first`, table keys, cell-wise equal labels) find 0 problems. Gate and report item statuses are unchanged; only depth-derived metrics move: those above and, on proseg_hybrid, the Neurons depth-gradient Spearman (0.515 to 0.552).
- **Outputs and status.** `$A/m13/followups_20261013/hd1_if_depth/` (`aug/compute_cortical_depth_out/`, `reports_aug/<segmentation>/`, `depth_comparison.json`, `report_comparison_reports_aug.json`, `check_structure_aug.txt`, `README.txt`). They are evidence only: the published HD1_IF results and `acceptance/` still hold the July-based outputs, and replacing them has not been done.

**9. Open after this record.** The user's ruling on the NP7 proposal (item 5) and on the NR8 proposal (item 6); restoring the WHB bundle `0e32a8f6` in the SSD1 store (item 6); whether the published HD1_IF results are replaced by the August-based outputs (item 8); the human Prime 5K and Xenium Human Brain (266) gate-P runs, a later step (item 2).

## 24. Mouse AP-axis rule selection and MO10 (M6b; pre-registered 2026-09-29, before any scoring run)

*Numbering (editorial, recorded 2026-10-13 at the merge of the M6b branch with the integration branch; no rule, threshold, reading or result in this section changes).* This section was written, committed and scored as **§17** on the M6b branch (commits up to `58ea185`), and the M6b evidence under `$A/m6b/` (scripts, `selection.json`, `STAGE_C_REPORT.txt`, the MO10 output) and the stage reports call it §17. The integration branch had meanwhile recorded the H12 section as §17 (and §18 to §23 after it), so at the merge this section became **§24**, the next free number; H12's §17 is untouched. Wherever the M6b evidence says "§17" it means this section.

Plan §7.8 ("M6b design (pre-registered now)"), §12 M6b, OD-C8 and MO10 (§9). This section copies the M6b design and fixes the operational details the plan leaves open **before** any variant or MO10 value is computed. No §9 threshold, target or decision rule changes. At the time of writing only the inputs exist: the MERFISH-C57BL6J-638850 files, the three panel bundles and the unpruned mappings (M6b stage A; `$A/m6b/`); no region inference, drop list, harm or T2 value has been computed on them.

**Selection rule (plan §7.8 item 2, OD-C8, verbatim):** "lowest maximum in-scope (direct) harm, subject to MO2 on [MS1] and [MS2] and T2 ≤ 0.1% in every bin; ties → the simpler rule; recorded in the preregistration before MO10 is scored." The rule picks among variants (a)–(d) and (f); (e) is withdrawn (the four Multiome-only WMB subclasses stay a documented limitation, nothing is filled in).

- **Data.** The public ABC MERFISH-C57BL6J-638850 raw expression h5ad (release 20230830; md5 `9484a3992a4b6d3a7c0281c72e1be522`, sha256 `f412b0f5…`) and the CCF cell metadata (`cell_metadata_with_parcellation_annotation.csv`, 20231215, the pinned `MERFISH_CCF_METADATA_PIN`), in `$MERXEN_STORAGE_ROOT/abc_atlas/MERFISH-C57BL6J-638850/` (`SHA256SUMS`, `README.txt`; CC BY-NC 4.0). Allen labels (class, subclass), `average_correlation_score`, section coordinates (`x_section`, `y_section`, mm) and the CCF parcellation come from the CCF view.
- **Sections and AP bins.** The 38 in-scope sections of `plan_rev2/ap_axis/section_region_share.csv` (CB + OB < 50% of grey cells; AP = median `x_ccf` of the section's cells; .61 at 2.42 mm to .17 at 10.37 mm). Bins as in plan §7.8: [2.4, 3.0), [3.0, 4.5), [4.5, 6.0), [6.0, 7.5), [7.5, 9.0), [9.0, 10.4] mm with 3, 6, 7, 8, 6 and 8 sections. The 15 out-of-scope sections are mapped for the AP estimate (item 4) only if needed there; they never enter MO10 or the selection.
- **Subsamples.** Per section, the cells of the CCF view that are also in the expression h5ad, sorted by `cell_label`; 30,000 drawn without replacement by `numpy.random.default_rng([0, NN])` (NN = the section number, e.g. 57 for `C57BL6J-638850.57`), or all cells when fewer. The same cells are used for the three gene sets. The cell lists and their sha256 are recorded in `$A/m6b/subsamples/`.
- **Gene sets.** MERFISH 500 (the 500 genes of the MERFISH `gene.csv`, Blank controls excluded), MERFISH ∩ MS1 (188) and MERFISH ∩ MS2 (346), intersected on Ensembl IDs with the resolved MS1 and MS2 panels (`panel_genes.json` of the M6 MAP outputs; panel hashes `f4c29186…` and `2a224a8e…`). By OD-E1 these mappings serve only the AP-axis rule; they do not validate a panel.
- **Mapping (production configuration).** One `wmb_panel` bundle per gene set (builder as on the branch, 10Xv3-only N = 50 marker precompute, Allen means `precomputed_stats_ABC_revision_230821.h5`, the MS1 ∪ MS2 marker universe for the two intersection sets and the panel's own genes for MERFISH 500, as §7.1 prescribes), new build dirs in the SSD1 store. Each section × gene set is mapped with `merxen annotate --species mouse --prepared-dir … --mouse-section-regions none` (drop SUPT, bootstrap factor 0.5, 100 iterations, seed 0, raw normalisation, `cell_type_mapper` 1.7.2, table cells ≥ 10 counts on the gene set). `none` keeps MAP unpruned; the region step is then applied by `scripts/acceptance/mouse_ap_axis.py` with the leave-neighbours-out tables below and each candidate rule, and the pruned re-map is the production one (`mouse_regions`: only the cells whose unpruned class or subclass is dropped are re-mapped, with `--nodes_to_drop` and the pruned lookup of the section's bundle).
- **Leave-neighbours-out tables.** For section s, the region tables (`reference.region_share_tables`: class and subclass shares of grey cells per region, homes, MERFISH cell numbers) are recomputed from the CCF view without the cells of s and of its immediate neighbours in the AP order of all 53 sections of the view (± 1 section; one neighbour at either end). Everything the rule reads from MERFISH comes from them: division homes, the structure-level AON homes of (a), the two-tier shares and cell numbers, and T2's shares. MS1 and MS2 (not MERFISH sections) use the full tables, i.e. the store's `wmb_region_share` bundle, as in production.
- **Confident neurons** (inference): table cells with a neuronal WMB class and class and subclass bootstrap probability ≥ 0.9 of the unpruned mapping (`mouse_regions`, unchanged); tiles on `x_section`, `y_section` × 1000 (µm).
- **Harm (direct; MO10 and the selection).** Per section and gene set: the number of table cells with `average_correlation_score ≥ 0.6`, class bootstrap probability ≥ 0.9 on the unpruned MERFISH mapping and an **Allen** class or subclass in the drop list, divided by the section's table cells. This is the computation behind the plan's basis values ("confident Allen cells … whose class or subclass is in the production drop list", `plan_rev2/review_generality/prod_drop.py`: in-drop confident cells over all cells of the section), with our class bp added as the plan asks. The same count over the confident Allen-labelled cells (both conditions) is reported beside it; it is larger by construction and is not used for MO10 or the selection. Extra regions are neutral. **Oracle-relative harm** (diagnostic only; E7 `09_autoregion_v2.py`): table cells whose Allen class or subclass is in nodes dropped under the inferred regions but not under the CCF regions holding > 0.2% of the section's grey cells (the same drop rule and variant applied to those regions), over table cells.
- **Also reported** per section, gene set and candidate: inferred regions; CCF regions (≥ 1% of grey cells); missed (CCF − inferred) and extra (inferred − CCF > 0.2%) regions; nodes dropped; relabelled (re-mapped) cells; T2 after pruning; each per AP bin (mean, maximum, sections above 0.1%).
- **T2 after pruning** is MO2's definition (`mouse_gate.t2_share`, E7: subclasses with ≥ 20 MERFISH grey cells and < 25% of them in the present regions, never-drop classes excluded), on the pruned subclass calls, with the section's present regions under the candidate and its leave-neighbours-out shares. "T2 ≤ 0.1% in every bin": for every gene set and AP bin, the mean over the bin's sections of the per-section T2 share is ≤ 0.1%; the per-section maximum is reported.
- **Candidates.** Every combination of the components (a) coupling `{"OB": ["OLF"]}`: OB is present when OLF is present and AON-home tiles are ≥ 1% of the assigned tiles, where a tile is AON-home when its majority structure-level home (the `parcellation_structure` holding the largest share of a subclass's grey cells, voted by the same confident neurons with the same tile, neuron minimum and tie rules) is `AON`; (b) never drop class `05 OB-IMN GABA` (with its subclasses) when STR or OLF is present; (c) never drop subclass `025 CA2-FC-IG Glut` when Isocortex and STR are present; (d) OB is present at ≥ 0.5% of the assigned tiles (other regions keep 1%; the component rule is unchanged), crossed with the tile settings (f) (`tile_um`, `min_neurons_per_tile`) ∈ {(150, 3), (100, 3), (200, 3), (150, 2), (100, 2), (200, 2)}, except the unchanged v1 rule itself: 95 candidates. v1 is scored as the reference; it is not a candidate (OD-C8: the rule picks among (a)–(d), (f)). Presence is decided first (with (d) for OB), then the coupling (a), then the two-tier drop rule with (b) and (c). Every other rule parameter stays at its M6 value (`MouseRegionConfig`).
- **Selection statistic.** S = the maximum harm over the 38 in-scope sections and the three gene sets. **Constraints:** (i) MO2 on MS1 proseg_hybrid and MS2 original_seg (the M6 MAP outputs, `$A/m6/stageC/map`): T2 after pruning ≤ 0.1%, class changes outside dropped nodes ≤ 0.25% and subclass ≤ 1.5% (subset re-map, production); (ii) T2 ≤ 0.1% in every bin for every gene set. The selected candidate has the lowest S among the candidates meeting both. Candidates may be evaluated in increasing S and the constraints checked until the first candidate that meets them and every candidate tying with it; this selects the same candidate as checking all.
- **Ties** are equal values of S (exact; per-section harms are cell-count ratios). The simpler rule is, in order: (1) fewer changes from v1 (each of (a)–(d) counts 1, each changed tile parameter counts 1); (2) fewer (section, gene set, node) drop decisions that differ from v1 over the 38 sections and three gene sets; (3) the alphabetical order of the canonical label (e.g. `a+b`, `a+b+f:tile100`).
- **No admissible candidate.** Nothing is selected; v1 stays, MO10 is scored on v1 and reported, and the choice goes to the user.
- **Order of record.** The candidate table and the selection are written to `$A/m6b/` and the selected candidate into this section (a dated commit) **before** MO10 is scored on it. MO10 is then scored per gene set exactly as §9 states (per-bin mean harm ≤ 0.10%, per-section maximum ≤ 0.30%, T2 after pruning ≤ 0.1% in every bin, missed regions only where that section's harm ≤ 0.1%, oracle-relative harm reported); it passes when all three gene sets pass. MO2 and MO3 on MS1 and MS2 are re-checked with the selected rule (M6b exit); MO3 is not a selection constraint.
- **Held out.** The MERFISH AP series stays held out except through this selection rule (§1 rule 3); the result is reported separately.
- **Interpretations recorded here** (the plan leaves them open; each keeps the §9 values): the harm denominator (table cells, as the basis values; confident-cell denominator reported); the bin statistic for T2 (the mean, as for harm); S as the maximum over the three gene sets; combinations of components and the (f) grid as candidates; v1 as the reference, not a candidate; the simplicity order; MAP unpruned with the region step in the scoring script.

**Operational details fixed at M6b stage B** (2026-09-29, with the scoring code and **before** any region inference, drop list, harm or T2 value was computed on the MERFISH mappings; only the CCF-view count tables exist, which reproduce the store's `wmb_region_share` tables exactly; no §9 value, threshold or decision rule changes):

- **(b) and (c) in T2** (amended 2026-09-29 17:50, see the note at the end of this list). A conditional never-drop node is kept by the drop rule in the sections where its condition holds on the present regions (a kept class with all its subclasses; a kept subclass also stops its class being dropped as one node). T2 after pruning keeps MO2's definition verbatim: only the static never-drop classes (`never_drop_classes`, E7's `NEVER_DROP`) are left out, so the calls a keep retains are counted when their subclass is T2 (`mouse_gate.t2_share`; the same for G3 and MO2 in production). T2 without the kept nodes (`mouse_regions.effective_never_drop`) is reported beside it as `t2_after_kept_excluded` and is not used.
- **(a) evidence.** AON-home tiles are the assigned tiles (the division-level tile rule) whose majority structure-level home, voted by the same confident neurons with the same tile size, neuron minimum and tie rule (first in sorted order), is `AON`; structure homes come from the same (leave-neighbours-out) cells as the division tables, and only grey-matter cells count. Coupling is one pass after the tile rule (a division added by coupling couples nothing further) and has no component condition; `coupling_min_tile_fraction` is 0.01 (>=).
- **(d)** is a per-division `min_tile_fraction` of 0.005 for OB only (>=, as the 1% rule); the component rule is unchanged.
- **Skipped sections.** A section with fewer than 200 assigned tiles is skipped as in production (no drop list, harm 0); its T2 after pruning is computed on its calls against the divisions the inference found, so a rule that stops pruning an in-scope section is not rewarded, and the skipped sections are counted per candidate.
- **CCF regions** of a MERFISH section (>= 1% for missed, > 0.2% for extra and the oracle) come from all its grey-matter cells in the CCF view, not only the subsample. **Oracle-relative harm** uses E7's node-name set difference (`09_autoregion_v2.py`: names of the nodes dropped under the inferred regions that are not dropped under the CCF regions). **"Sections above 0.1%"** means > 0.1%.
- **Relabelled cells** are the re-mapped cells (unpruned class or subclass dropped), with the class and subclass changes among them reported. Re-maps are run once per distinct drop list of a section and gene set (identical inputs give the production re-map of every candidate with that list), with the MAP run's own engine parameters **including its process count** (6): `cell_type_mapper`'s bootstrap draws depend on it (amended 2026-09-29 18:00: the first version said 2 processes; the smoke check `$A/m6b/stageB/remap_nproc_check/COMPARE.txt` found a 2-process re-map of MS1's v1 drop list differing from M6's production re-map on 10.2% of class labels, and the 6-process re-map identical on all 394 cells and four levels).
- **Order of record, operationally:** `scripts/acceptance/mouse_ap_axis.py mo10` refuses to run unless this section contains the line ``Selected candidate (M6b): `<label>` `` (added in the dated selection commit).
- **AP estimate** (plan §7.8 item 4; not a selection or MO10 input): the query is the section's confident neurons as the region rule defines them (neuronal class, class and subclass bootstrap probability >= 0.9; pruned calls in RESOLVE), non-neuronal subclass names (` NN`) ignored, at least 100 cells; the reference is each MERFISH-638850 section's neuronal subclass counts (`section_composition.parquet` of the `wmb_region_share` bundle); the estimate is the AP of the section with the highest Bhattacharyya coefficient (a top-5 softmax-weighted AP, temperature 0.01, is reported beside it); G4's window is that section and its ± 1 neighbours among all 53 sections. Held-out validation (reported, no gate): each section left out of the reference, pass when the error is at most the section's step (its larger AP gap to an immediate neighbour); on the Allen labels (neurons with `average_correlation_score >= 0.5`) of all 53 sections and on our mappings of the 38 in-scope sections for the three gene sets.
- **Amendment note (2026-09-29 17:50).** The first version of this list (commit 3920583) excluded the kept nodes from T2. That departs from the pre-registered T2 ("MO2's definition ..., never-drop classes excluded", which at pre-registration was the static list the M6 code passes), so it was reverted to the literal definition before any T2-after-pruning value was computed: at that point only a plumbing smoke run of `plan` (two sections of the MERFISH ∩ MS1 set, .57 and .61, and MS1 / MS2) existed, under `$A/m6b/stageB/smoke/` (not a scoring run; stage C re-plans every section). The change can only make candidates with (b) or (c) less admissible.

**Stage C result (recorded 2026-10-13, before MO10 is scored).** The M6b scoring finished on 2026-09-29 (95 candidates and v1 over 38 sections and three gene sets; code `b8f8786`, whose code files are byte-identical to this branch's head before this commit). The outputs are `$A/m6b/scoring/summary/{selection.json,candidates.csv,CANDIDATES.txt,bins.csv,sections.csv,real.csv}`. This is the selection outcome only; no MO10 value has been computed or read for this record.

- `selection.json`: `"selected": null`, `"complete": true`, reason "no admissible candidate: v1 stays and the choice goes to the user".
- Every one of the 95 candidates, and v1, fails the T2 constraint (ii): for each of the 96 rules the largest bin-mean T2 after pruning (over the three gene sets and six AP bins) is above 0.1%. The range over the 96 rules is 0.388% to 0.907%; v1's is 0.399% (`candidates.csv`, column `max_t2_bin_mean`; `t2_ok` is False in all 96 rows).
- MO2 (constraint (i)) passes on MS1 proseg_hybrid and on MS2 original_seg for all 96 rules (`mo2_ok` is True in all 96 rows). No candidate is admissible (`admissible` is False in all 96 rows).
- The lowest maximum direct harm S is 0.210% (`a+f:tile200,nn2` and its ties); v1's S is 0.757%. S is not used: this rule selects only among candidates that meet both constraints.
- v1 stays. Per "No admissible candidate" above, MO10 is scored on v1 and reported, and the choice of rule goes to the user. Nothing in this subsection changes a §9 threshold, a target or a registered reading; any change to the T2 reading or to MO10 needs the user's written approval (§1 rule 2) and is recorded as a dated amendment of this section.
- Diagnostics of the T2 metric itself (the Allen labels scored with the same regions, the T2 calls' agreement with the Allen labels) are in `$A/m6b/STAGE_C_REPORT.txt`. They are not scores and no selection or MO10 value depends on them.

Selected candidate (M6b): `v1`

### Amendment of 2026-10-14 (U1): the T2 reading approved by the user

*Recorded 2026-10-14. At this point the C1 diagnostics (`$A/m6b/STAGE_C_REPORT.txt`, `$A/m6b/stageC/t2diag/`) and MO10 scored on v1 (`$A/m6b/scoring/mo10/mo10.json`, FAIL) had been read. The selection has not been re-run under this reading, and MO10 has not been scored on any rule other than v1.* The quotations below are verbatim. Their source is `$MERXEN_ANNOTATION_DEV_DIR/mouse_roadmap_20261013/USER_DECISIONS_20261014.md`, which was extracted from the session transcript.

**The question** (put in the chat on 2026-10-14 at 08:12 UTC):

> 1. **M6b rule (U1).** Even a perfect annotation fails "T2 ≤ 0.1% in every bin": the Allen labels themselves fail 5 of 6 bins.
>    - **Recommendation:** read the bound as "T2 no higher than the Allen labels' own T2, with the same regions". That selects rule `a+f:tile200,nn2` over v1. Gate M would still need one written MO10 exception.
>    - This is a post-hoc loosening, so it needs your written approval.

**The user's answer** (chat, 2026-10-14 at 08:20 UTC): "1. go with recommendation".

**Label: LOOSENING, POST-HOC, approved in writing on 2026-10-14** (§1 rule 2). The gate M PR quotes the question and the answer.
- *Loosening.* The Allen-label T2 bin means of the three diagnosed rules are 0.084–0.618% (STAGE_C_REPORT §2), so this bound admits rules that the 0.1% bound excluded. In one bin it is stricter than 0.1%: in 2.4–3.0 mm the Allen-label T2 is 0.084%.
- *Post-hoc.* Before this record, C1 computed the reading for v1, `a+f:tile200,nn2` and `f:tile200,nn2`. Each meets it in all 18 (gene set × bin) cells (STAGE_C_REPORT §2 (c); `T2_DIAGNOSTIC.txt`). The effect on the selection is therefore known (see "Expected selection" below). The MO10 parts of `a+f:tile200,nn2` are also known from the candidate table ("roadmap" below is the mouse roadmap of 2026-10-13, `annotation_dev/mouse_roadmap_20261013/MOUSE_ROADMAP.md`; here its §1).

**The approved reading, operationally.** For a rule R (v1 or any of the 95 candidates), a gene set g and an AP bin b:

1. For each in-scope MERFISH section s of bin b, take:
   - *the cells:* the table cells of s on g, the cells the registered T2 after pruning counts (n = `n_table_cells`);
   - *the T2 subclass set* of the registered T2 after pruning for R on (g, s): `mouse_gate.t2_subclasses(regions, shares, subclass_class=…, never_drop_classes=…, min_merfish_cells=…)`, the subclasses with ≥ 20 MERFISH grey cells and < 25% of them in the regions. Its inputs:
     - `regions` is R's `t2_regions` for (g, s): the present regions R infers, or, for a skipped section, the divisions the inference found;
     - `shares` are the leave-neighbours-out shares of s;
     - the never-drop classes are the static ones of R's configuration, and the cell minimum is its `min_merfish_cells_subclass`.
2. `t2_after(R, g, s)` is the share of those cells whose pruned subclass call (after R's re-map) is in the set. This is the registered value.
3. `t2_allen(R, g, s)` is the share of the same cells whose **Allen subclass label** is in the same set. Every table cell counts: no `average_correlation_score` or bootstrap-probability filter applies.
4. The bin means `T2_after(R, g, b)` and `T2_allen(R, g, b)` are the unweighted means over the bin's sections, as for the registered bin mean.
5. **R meets the bound** when `T2_after(R, g, b) ≤ T2_allen(R, g, b)` for every gene set g and every AP bin b.
   - The values are compared as fractions in float64, with an absolute tolerance of 1e-12 for rounding only. The diagnostic and the scoring differ by at most 6.8e-17 (STAGE_C_REPORT §3).
   - A missing value on either side fails.

These are exactly the columns `t2_after` and `t2_allen_labels` of `$A/m6b/stageC/t2_diagnostic.py`: per section in `t2diag/t2_sections.csv`, and as bin means in `T2_DIAGNOSTIC.txt`. The class and subclass taxonomy comes from the verified storage-RAID bundle, as in STAGE_C_REPORT §0; the SSD1 M6b bundles are never read.

**What the reading replaces.**
- **Selection constraint (ii).** "T2 ≤ 0.1% in every bin for every gene set" becomes "R meets the bound above". Nothing else in the selection changes:
  - constraint (i), MO2 on MS1 proseg_hybrid and MS2 original_seg: their own T2 after pruning ≤ 0.1%, class changes ≤ 0.25% and subclass changes ≤ 1.5% outside dropped nodes, on the subset re-map;
  - the statistic S, the tie rule and the "no admissible candidate" path;
  - the order of record;
  - the inputs: the scoring outputs of 2026-09-29 in `$A/m6b/scoring`, with no new re-map.
- **MO10's T2 clause. Reading adopted: "both". Label: LOOSENING, POST-HOC** (against the alternative below).
  - *Loosening:* MO10's T2 clause registers 0.1% in every bin. The Allen-relative bound admits higher values wherever the Allen labels' own T2 is above 0.1%, which the question put at 5 of 6 bins (see *Loosening* above).
  - *Post-hoc:* when it was adopted, C1 had shown the three diagnosed rules meeting the Allen-relative bound, and `candidates.csv` showed every rule failing the 0.1% bound.
  - *Approval:* the orchestrator reads it as within the written approval of 2026-10-14, for the reasons in the next two bullets. Also, the exception that approval covers is the missed-region clause only (below); under the alternative, the exception would have to cover the T2 clause as well, which the approval did not. The gate M PR states this reading beside the quoted answer.
  - The quoted bound, "T2 ≤ 0.1% in every bin", is selection constraint (ii) (the selection rule of this section). It is also MO10's T2 clause (§9: "T2 after pruning ≤ 0.1% in every bin").
  - The question asked to read "the bound" as the Allen-relative one, and did not restrict that to the selection. MO10's T2 clause is therefore scored with the bound above too, per gene set as this section states.
  - *Alternative, not adopted:* the reading replaces constraint (ii) only, and MO10's T2 clause keeps 0.1%. STAGE_C_REPORT §6 recommended this alternative. Under it the T2 clause fails for every rule, and the written MO10 exception would have to cover it as well.
  - MO10's T2 is reported under both readings: the 0.1% verdict and the Allen-relative verdict per gene set and bin. The Allen-relative verdict is the scored one.
- **Not changed:** MO2's own T2 bound on MS1 and MS2 (0.1%), in constraint (i) and at gate M.

**Expected selection: `a+f:tile200,nn2`** (from the C1 diagnostics).
- It meets the bound (18/18) and MO2 (STAGE_C_REPORT §1, §4).
- No candidate has a lower S than its 0.210%.
- Of the 14 rules tied at that S, it comes first in the tie order. It has 3 changes, tying `d+f:tile200,nn2` and `b+f:tile200,nn2`, and 662 differing drop decisions, against their 694 and 990.

This is why the amendment is POST-HOC.

**How M6b-R applies it.**
1. It implements the reading in `scripts/acceptance/mouse_ap_axis.py summarise`: the Allen-label T2 per section beside `t2_after`, its bin means in `bins.csv`, and the registered 0.1% check kept as a reported column.
2. It computes the reading for all 96 rules, so that the candidate table is complete.
3. It re-runs the selection on the scoring outputs of 2026-09-29.
4. It records the result in this section as a new dated line naming the selected candidate. That is the line the `mo10` guard reads. The v1 line above records the outcome of 2026-10-13 and stays. This amendment adds no selection line.
5. Only then does it score MO10 on the selected rule and re-check MO2 and MO3 on MS1 and MS2.

**Gate M carries one written MO10 exception** (U1). The gate M PR gives its text and numbers.
- *Scope:* the MO10 parts that fail on the selected rule as M6b-R scores it, listed in the gate M PR.
- *What the approval covers:* the parts expected when the question was put, the missed-region clause at two sections (roadmap §1):
  - .35 (AP 7.6 mm): STR missed, harm 0.13–0.14%;
  - .46 (AP 5.7 mm): CTXsp missed, harm 0.103%.
- *Anything further goes back to the user* before the gate M PR. That means any other MO10 part that fails: a bin-mean harm, a section-maximum harm, the T2 clause under the adopted reading, or a missed region at another section.
- MO11 and its banner are unchanged.

**Selection under the approved reading (recorded 2026-10-14, before MO10 is scored on the selected rule).**
- *Run.* `scripts/acceptance/mouse_ap_axis.py summarise --t2-reading allen-relative` at `297f084`, into the new directory `$A/m6b/scoring_u1`. Its inputs are the 2026-09-29 scoring's tables, plan rows and re-maps, linked read-only from `$A/m6b/scoring`; no re-map ran. The `$A/m6b/scoring` files have the same sha256 before and after the run (2,312 files). The taxonomy came from the verified storage-RAID bundle `af3fab09…` (`--taxonomy-bundle`, checked against every call and Allen label), and the region tables of MS1 and MS2 from the storage-RAID bundle `1867f978…`, whose files are byte-identical to those of the scoring's bundle. No SSD1 path was opened (`scoring_u1/runner.py`, `run_u1.sh`, `PROGRESS.txt`).
- *Reproduction* (`scoring_u1/REPRODUCTION.txt`). Every column of the 2026-09-29 `sections.parquet`, `real.csv`, `bins.csv` and `candidates.csv` is reproduced exactly; `t2_ok_registered` equals the old `t2_ok` in 96/96 rows. The Allen-label T2 equals the stage C diagnostic's `t2_allen_labels` exactly in 342/342 section rows, and the 54 bin-mean rows of `T2_DIAGNOSTIC.txt` are reproduced as printed.
- *Outcome* (`scoring_u1/summary/selection.json`, `candidates.csv`, `CANDIDATES.txt`).
  - 43 of the 96 rules meet the approved bound, v1 among them; none meets the 0.1% bound; all 96 meet MO2. So 42 candidates are admissible.
  - The lowest S among them is 0.210%, shared by 7 candidates. The 7 other rules at that S, all with component (c), exceed the Allen labels' T2 in one bin (merfish_ms1, 3.0–4.5 mm: 0.4067% against 0.4061%).
  - The tie order selects `a+f:tile200,nn2`: 3 changes, tying `b+f:tile200,nn2` and `d+f:tile200,nn2`, and 662 differing drop decisions against their 990 and 694. This is the expected selection above.
- MO10 has not been scored on this rule. It is scored next, with `mo10 --out $A/m6b/scoring_u1`, whose guard reads the line below.

Selected candidate (M6b): `a+f:tile200,nn2`

**MO10 on `a+f:tile200,nn2` (result, recorded 2026-10-14 after scoring; nothing here changes a reading, threshold or rule).** Scored at 11:59 with `mo10 --out $A/m6b/scoring_u1`, after the selection commit `35455e8` (11:18). The outputs are in `$A/m6b/scoring_u1/mo10/` (`mo10.json`, `MO10.txt`).
- *Adopted reading (Allen-relative T2): FAIL.* Only the missed-region clause fails, at two sections:
  - .35 (AP 7.6 mm), STR missed: harm 0.140% in MERFISH 500 and 0.133% in MERFISH ∩ MS2;
  - .46 (AP 5.7 mm), CTXsp missed: harm 0.103% in MERFISH 500.
  MERFISH ∩ MS1 passes. In all three gene sets the bin-mean harm (at most 0.080%), the section maximum (at most 0.210%) and the T2 clause pass.
- These are the parts the approval covers ("What the approval covers" above), so the written MO10 exception of gate M has this scope and nothing goes back to the user under U1.
- *Registered T2 reading (0.1%), reported:* the T2 clause also fails, in 6 of the 18 (gene set × bin) cells.
- *MO2 and MO3 with the selected rule* pass on MS1 proseg_hybrid and MS2 original_seg (`scoring_u1/summary/real.csv`; STAGE_C_REPORT §4). The production region step was also run with the default rule and the revision-2 `wmb_region_share` bundle `2bf9c01c…`, with no re-map, on the M6 MAP output of MS1 and run3's MAP output of MS2. It infers the scoring's regions and drop list on both, which is v1's recorded drop list (`$A/m6b/region_share_r2/mo2_mo3/CHECK_REGIONS.txt`).

## 25. Gate M scoring protocol (GM-PREREG; recorded 2026-10-14, before any gate M score is read)

Gate M (plan §12 M9) scores the mouse criteria of §9 on MS1 and MS2. This section fixes how, before any gate M run has produced a score. No gate M run has started: GM-RUNS starts after M6b-R, C22, A1 and the scorer (GM-SCORER). No §9 threshold, target or decision rule changes here, except through §24's amendment (U1).

The sources are:
- the user's answers of 2026-10-14, verbatim in `annotation_dev/mouse_roadmap_20261013/USER_DECISIONS_20261014.md`;
- the roadmap's U4 readings (`annotation_dev/mouse_roadmap_20261013/MOUSE_ROADMAP.md` §3);
- G0's memory probe (`$A/m9/mo8_probe/RESULT.txt`).

The labels follow roadmap §3: tightening, neutral, LOOSENING, POST-HOC.

### 25.1 MO8 (U4)

**The question** (chat, 2026-10-14 at 08:12 UTC, verbatim):

> 2. **Gate M memory limit (U4, MO8).**
>    - The [MS2] mapping peaks at 14.0 GB with 6 processes, against a 12 GB limit. With 4 processes it peaks at 11.0 GB and stays within the time limit, at 0.60 h against 1.25 h.
>    - The catch: with 4 processes, 293 of 2,848 cells in [MS2]'s region-step re-map get a different assignment. MapMyCells' bootstrap evidently depends on the process count.
>    - **Recommendation:** run gate M's mapping with 4 processes, which meets the registered reading with no loosening, and record the process-count dependence as a known reproducibility caveat.

**The user's answer** (chat, 2026-10-14 at 08:20 UTC, verbatim): "2. go with recommendation, its ok to use more RAM if needed".

**Correction of fact.** The question's "293 of 2,848 cells" are 293 of 2,848 *level rows*: 712 re-mapped cells × 4 levels. Of the 712 cells, 29 change at class, 72 at subclass, 81 at supertype and 111 at cluster (RESULT.txt). The recommendation does not depend on this.

**The reading: H14's, as registered (neutral).**
- **MAP peak ≤ 12 GB.**
  - Per section, the value is the largest `peak_rss` of its MAP tasks (`CLUSTERING_SQUIDPY_ANNOTATE_MAP`) over the four segmentations, from the Nextflow trace (GB = 1024³ bytes).
  - This is gate H's reading of H14 (§18, "Stage B scoring as implemented", H14; `scripts/acceptance/run_acceptance.py` `h14_rows`).
  - Nextflow's `peak_rss` is the peak over time of the summed VmHWM of the task's process tree (RESULT.txt, note 1).
- **Gate M's MAP runs with 4 MapMyCells processes.**
  - The MAP task's `cpus` is set to 4; it is passed as `--n-processors`, and the default is 6. MO7's standalone run maps with 4 processes too.
  - If a MAP task of a scored section ran with another count (the trace's `cpus` or the MAP manifest), that section's MO8 rows are not scored and go back to the user.
- **Time ≤ 1.25 h per 4-segmentation section.** This is H14's sum: the realtime of ANNOTATE_PANEL + MAP + RESOLVE over the section's four segmentations, from the trace (`h14_rows`).
- **PREP ≤ 2 h per panel, resolvability included.**
  - Measured as gate H measured it (§18, "Stage B scoring as implemented", H14: "per PREP bundle, its recorded build time incl. resolvability"; `h14_rows`, row `H14/prep_wall`): each `wmb_panel` bundle's recorded `wall_time_s` in its `bundle.json`.
  - The gate M bundles are MS1 `97a086d1…` (built in B1) and MS2 `af3fab09…`. The `wmb_region_share` and `wmb_selfmap_testset` builds are reported beside them.
- **PREP peak ≤ 40 GB at 8 processes.**
  - *Precedent:* gate H scored no PREP memory, because H14 has no such clause. Its memory reading, the trace `peak_rss`, is the summed memory of a task's process tree. The same reading for PREP is therefore the summed RSS of the build's process tree.
  - *Scored value:* the larger of two values:
    - the largest `peak_tree_rss_gb` over the ctm steps recorded in the bundle's `bundle.json` (`builder_output.ctm_steps`; `memory.ProcessTreeSampler`, GiB, sampled every 2 s);
    - where the gate M run itself built the bundle, the trace `peak_rss` of that PREP task.
  - *8 processes:* the bundle must have been built with 8 processes (`--n-processors 8`; its resolvability `mapping_runs` record 8).
  - *Reported beside it:* the tree PSS (`peak_tree_pss_gb`), and the largest single process (`peak_rss_gb`, from GNU time). The single process is how MO8's basis, "query markers measured 20–27 GB at 16", was measured.
  - Of these three readings, the scored one is the strictest.

**G0's numbers** (MS2 original_seg MAP: run3's task, re-run; RESULT.txt). These were seen before this record.

| Processes | Trace-style peak (GiB) | Tree PSS (GiB) | MAP wall (min) | Projected H14 sum over 4 segmentations (h) |
|---|---|---|---|---|
| 6 | 13.97 (fails 12) | 7.29 | 10.27 | 0.758 (0.827 scaled by run3's ratio) |
| 4 | 11.03 (passes, margin 1.0) | 5.64 | 7.85 | 0.597 (0.649 scaled) |

- *run3's trace:* MAP `peak_rss` 13.99–14.72 GiB at 6 processes, and an H14 sum of 0.980 h.
- *The largest segmentation at 4 processes:* projected at 11.6 GiB by run3's largest-to-original_seg ratio. This is an estimate.
- *Host load:* the 6-process run overlapped a PREP (load1 mean 32), and the 4-process run had a quieter host (15). The wall times are therefore not comparable. The memory peaks do not depend on the load.

**Reproducibility caveat (recorded with U4).** MapMyCells' bootstrap depends on the process count.
- *What G0 found:* at 4 processes, 293 of the 2,848 level rows of MS2 original_seg's region-step re-map differ from the 6-process run. The 112,158-cell main map is identical, and a 6-process re-run reproduces run3 exactly (RESULT.txt).
- *Consequence for the baseline:* gate M's labels of the re-mapped cells are those of its 4-process runs. M6's and run3's 6-process outputs are not their baseline, and every gate M criterion is scored on gate M's own outputs.
- *Consequence for the flip:* MO8's verdict holds for the configuration it measures, 4 processes. The flip PR states the mouse MAP process count it ships, and a count other than 4 goes to the user at GM-REVIEW.

**PREP numbers seen before this record.** These come from the `bundle.json` of the two bundles. Roadmap §3 U4 gives MS2's tree RSS; the MS1 values were read while this section was written.

| Bundle | Wall | Summed tree RSS (query markers) | Tree PSS | Largest single process |
|---|---|---|---|---|
| MS2 `af3fab09…` | 3,245 s (0.90 h) | 56.80 GiB | 25.54 GiB | 25.84 GiB |
| MS1 `97a086d1…` | 3,101 s (0.86 h) | 47.70 GiB | 18.92 GiB | 19.26 GiB |

Under the reading above, PREP time passes and **PREP peak is expected to fail on both panels**. Under the tree-PSS reading or the single-process reading it would pass. Adopting either of those now would be a LOOSENING, POST-HOC (roadmap §3 U4 (b)).

**"its ok to use more RAM if needed".**
- The sentence is recorded verbatim.
- It is not read as a change to MO8, and it is not used to pass MO8. MO8 is scored under the registered reading above.
- If MO8 fails, as the PREP peak is expected to, GM-REVIEW asks the user whether that sentence is the written MO8 exception. The gate M PR quotes the answer.

#### 25.1.1 The written MO8 exception for the PREP peak (recorded 2026-10-14, before any gate M run)

**Names.** These afternoon records (§25.1.1, §25.2.1 and §25.3.1) name the two mouse sections by their aliases, under the privacy rule the user set on 2026-10-14 (`USER_DECISIONS_20261014.md` section F): **MS1** is the ageing-set section and **MS2** the Vizgen validation section, the two sections that §24–§26 name otherwise. Inside quotations, a name in square brackets is the alias standing for the original; the originals are kept in the private decisions file.

**The question** (chat, 2026-10-14 at 13:25 UTC, item 2, verbatim):

> 2. Gate M will probably fail MO8's reference-prep memory limit. Prep peaks at 48 GB for [MS1] and 57 GB for [MS2], against a 40 GB limit. At the gate M review I'll ask whether your "its ok to use more RAM if needed" counts as the written exception.

**The user's answer** (chat, 2026-10-14 at 13:38 UTC, verbatim): "2. You are allowed to use more RAM for those processes."

The source is `annotation_dev/mouse_roadmap_20261013/USER_DECISIONS_20261014.md` section E, which quotes the exchange verbatim.

**The reading: the written MO8 exception. LOOSENING, POST-HOC, approved in writing on 2026-10-14.**
- *What it covers.* Only MO8's part "PREP peak ≤ 40 GB at 8 processes", for the two gate M panels: the `wmb_panel` bundles `97a086d1…` (MS1) and `af3fab09…` (MS2) of §25.3. A scored PREP-peak row of either bundle that fails becomes `FAIL (excepted)`.
- *What it does not cover.* The answer does not cover:
  - MO8's MAP peak (≤ 12 GB, at 4 MapMyCells processes);
  - MO8's time limits (≤ 1.25 h per 4-segmentation section; PREP ≤ 2 h per panel);
  - the PREP peak of any other bundle.

  A failure there goes back to the user.
- *Post-hoc.* The peaks were known when the answer was given: 47.70 GiB (MS1) and 56.80 GiB (MS2) summed tree RSS (§25.1, "PREP numbers seen before this record"; the question rounds them to 48 and 57 GB).
- *The reading of the value is unchanged.* The PREP peak is still scored under §25.1's registered reading: the summed tree RSS, the strictest of the three readings, for a bundle built with 8 processes. The exception changes the verdict of the rows it covers, not their value. A bundle not built with 8 processes stays `UNSCORED (process count)`, which no exception reaches.
- *§25.1's open question is answered.* §25.1 left it to GM-REVIEW to ask whether "its ok to use more RAM if needed" is the written MO8 exception. The question was put before gate M instead, and this answer is the written exception for the PREP peak. The earlier sentence is still not read as a change to MO8.
- *In the scorer.* One `--exceptions` entry per panel: `criterion` `MO8`, `scope` `{"part": "prep_peak", "bundle": "<the panel's full build hash>"}`, quoting this answer. The gate M PR quotes the answer with the approval.

### 25.2 Readings adopted before scoring (orchestrator; labels as in roadmap §3 U4)

Each reading the orchestrator adopts here is labelled tightening or neutral. None of them is a loosening, so none needs written approval (§1 rule 2). The MO10 bullet is not one of them: it applies §24's amendment (U1), a LOOSENING, POST-HOC, approved in writing on 2026-10-14. The written MO10 exception that the flip-rule bullet names comes from the same approval. The gate M PR lists them all.
- **MO2 (tightening).** Both the production (subset) re-map and the full re-map must pass: T2 after pruning ≤ 0.1%, class changes outside dropped nodes ≤ 0.25% and subclass changes ≤ 1.5%.
  - Each re-map runs with the MAP run's own engine parameters, its 4 processes included, as §24's stage-B rule requires for re-maps.
  - At M6, MS1's full re-map was .245%, against the .25% limit.
- **MO4 (neutral).** The registered fixed window, MERFISH sections .31–.33, is scored for MS2 original_seg and for MS1, with §9 MO4's bands.
  - The AP-estimate window is reported beside it. For MS1 it is .32/.33/.35, with an Astro-Epen reference of 18.40%, against 17.36% for the fixed window (roadmap §3 U4). For MS2 it equals the fixed window.
- **MO7 (neutral).** The invalid MS2 proseg_hybrid is scored as a standalone run (`merxen annotate` and `merxen annotate-resolve` on the merged code).
  - *Input:* the legacy-snapshot clustered H5AD, `$MERXEN_LEGACY_SNAPSHOT_DIR/mouse_ms2_results/MS2/proseg_hybrid/clustering_squidpy/clustering_squidpy_out/merscope/MS2_MERSCOPE_clustered.h5ad`. These are the bytes M6 used.
  - *G1:* M6's G1 JSON, `$A/m6/stageC/registration/ms2_old_invalid_proseg_hybrid_ownrun_registration_qc.json`, as M6's MO7 run used it.
  - The full pipeline results of this run were deleted on 2026-09-30.
- **MO9 [L] (neutral, as registered).** Reported. It gates the use of the reseg and proseg outputs, not the flip.
- **H18 mouse rows (neutral).** Recorded with their values and no exception (§18, "Not covered", CHECK M2). They are not in the mouse flip rule. They set M13b's H18 set under D28 (§26).
- **MO10 (LOOSENING, POST-HOC, approved in writing on 2026-10-14; §24's amendment, U1; not an orchestrator reading).** Scored as §24's amendment states: the Allen-relative T2 reading, scored by M6b-R on the selected rule, with the one written MO10 exception.
- **The scored MS2 segmentation (neutral; added 2026-10-14, before any gate M run).** For MS2, every MO that names no segmentation (MO2, MO3, MO5, MO6) is scored on original_seg. §9's default "on proseg_hybrid unless stated" cannot apply to MS2: its proseg_hybrid is the invalid section that MO7 scores as the negative control, and MO1, MO4 and MO7 already name MS2 original_seg. M6's exit scored MS2 the same way. MS1 is scored on proseg_hybrid, as §9 registers.
  - *Corrected on 2026-10-14 (§25.2.1): the justification above is wrong, and the reading is replaced by a tightening.*
- **MO5's astrocyte FPR on MS1 too (tightening; added 2026-10-14, before any gate M run).** The astrocyte false-positive rate ≤ 0.5% must hold on MS1 proseg_hybrid as well as on MS2 original_seg, as M6's exit scored it. The ≥ 10× enrichment of held-out microglial genes stays a MS2 criterion, as registered.
- These two bullets were added after the scorer was written and its smoke test ran on run3's outputs (pre-merge code, labelled "smoke test, not gate M"; no verdict was read). No gate M run had started.
- **MO1, MO3, MO6 and MO11** are otherwise scored as §9 registers them.
- **The mouse flip rule is unchanged:** MO1–MO8 and MO10, with the one written MO10 exception of §24. MO11 keeps its banner rule. Any other failure needs a fix, or a written, user-approved exception in the gate M PR.
- **Left for GM-REVIEW** (roadmap §3):
  - U8: OD-B7's mouse half;
  - U9: the user's review of the MS1 and MS2 original_seg reports, and the written exceptions;
  - U10: the version label.

#### 25.2.1 Correction of 2026-10-14: the scored segmentation of MS2 (tightening; recorded before any gate M run)

This record corrects the bullet "The scored [MS2] segmentation" above, added in commit `c9486b9`. Names follow §25.1.1.

**What was wrong.** The bullet reads §9's default "on proseg_hybrid unless stated" as impossible for MS2, because "its proseg_hybrid is the invalid section that MO7 scores as the negative control". That justification is wrong.
- The invalid proseg_hybrid that MO7 scores is MS2's **old** proseg_hybrid. Its full pipeline results were deleted on 2026-09-30, and MO7 scores it from the legacy-snapshot clustered H5AD (§25.2, MO7).
- MS2's **current** proseg_hybrid is the output of the M0a re-run. It is a valid output, and §9's MO9 [L] gates its use, not the flip.
- §9's default therefore reads literally as MS2 proseg_hybrid for every criterion that names no segmentation.

**The corrected reading.**
- For MS2, every MO that names no segmentation is scored on **both** MS2 proseg_hybrid (§9's literal default) and MS2 original_seg (what M6's exit scored). Both must pass. The criteria are:
  - MO2: both the production (subset) re-map and the full re-map (§25.2, MO2), so MS2 proseg_hybrid gets its own full re-map, made with its MAP run's own engine parameters;
  - MO3;
  - MO5: the held-out microglial-gene enrichment and the astrocyte FPR;
  - MO6.
- MO5's astrocyte FPR is also scored on MS1 proseg_hybrid. The bullet on MO5's astrocyte FPR, also added in `c9486b9`, stands.
- MS1 is scored on proseg_hybrid, as §9 registers.
- MO1, MO4 and MO7 name MS2 original_seg and are scored as registered. MO8 is per section and per PREP bundle. MO9 [L] is reported.
- A scored row that cannot be measured on either MS2 segmentation is `NOT_AVAILABLE` and goes back to the user, as every scored row does.

**Label: tightening.** No approval is needed (§1 rule 2).
- Each criterion is scored on {proseg_hybrid, original_seg} for MS2, and all of them must pass.
- This is no easier to pass than the literal reading (proseg_hybrid alone) or the reading of `c9486b9` (original_seg alone).
- This record replaces the bullet's label (neutral) and its justification.

**Order of record.**
- No gate M run had started when this was recorded. The two gate M run directories were prepared but not launched; they hold staged inputs only, no gate M output.
- *The scorer's smoke test.* It ran on run3's outputs (pre-merge code, labelled "smoke test, not gate M"; it scored MS2 on original_seg). It computed MO3, MO5 and MO6 rows for MS2 proseg_hybrid as reported rows. It computed no MO2 row for MS2 proseg_hybrid, because MO2 was measured on the scored segmentation only. `c9486b9` records that no verdict of it was read.
  - For this record, only two things were looked at: the smoke test's row keys (criterion, dataset, segmentation), to establish which segmentations it covered, and the verdict totals its log prints over all rows.
  - No value or verdict of an MS2 proseg_hybrid row was read.
- *Disclosed prior knowledge.* M6's exit report (`$A/m6/M6_EXIT_REPORT.txt`, 2026-09-29) lists M6-era values of MS2's re-run proseg_hybrid beside the scored ones. Among them are MO3's inferred regions, MO5's two quantities and the F1 rate that MO6 bounds. They were seen at M6's review.
  - They come from M6's own MAP and RESOLVE outputs: another MAP process count, the v1 region rule and bundles that gate M does not use.
  - Gate M scores its own outputs (§25.1, reproducibility caveat), so none of them is a gate M value.

### 25.3 Inputs (U3)

- **Code.** Merged code only: the integration branch with PR #46, M6b (PR #52), M6b-R, C22 and the scorer.
- **Datasets.**
  - MS1: all four segmentations, run in place (below).
  - MS2: all four segmentations. RESOLVE and REPORT write into its results with the table suffix `gatem`, so run3's `_mapfirst` tables stay. MAP re-runs at 4 processes.
  - MO7's standalone negative control (§25.2).
- **Bundles.** Every bundle comes from the store `$MERXEN_REFERENCE_STORE`:
  - MS1: `wmb_panel` `97a086d1…`, panel hash `f4c29186…`;
  - MS2: `wmb_panel` `af3fab09…`;
  - `wmb_region_share` `1867f978…`, or the bundle that M6b-R builds if the selected rule needs component (a)'s structure homes;
    - *Replaced on 2026-10-14 by `2bf9c01c…`, revision 2 (§25.3.1).*
  - the self-map test sets of both panels.

  No bundle from SSD1 is used, and no `bundle_ref.json` that points at SSD1. The scorer's provenance check fails any other bundle path.

**MS1 runs in place (U3).**
- *The route:* the user chose "Run in place, suffixed". This was the answer, given 2026-10-14 at 08:09 UTC, to the question of 2026-10-13 at 16:46 UTC. The option text read: "Write suffixed _mapfirst tables into their store; needs their agreement first."
- *The owner's agreement:* asked "3. **[MS1] in place (U3).** Writing suffixed tables into [MS1]'s store needs [the MS1 data owner]'s agreement. Has she agreed?" (2026-10-14 at 08:12 UTC), the user answered "3. Yes she has" (08:20 UTC). In that quotation, "[the MS1 data owner]" replaces the owner's user name.
- *What may be written:* writes into the MS1 zarr store are limited to:
  - the suffixed `_mapfirst` tables;
  - the zarr root metadata entries for them;
  - the store's write-lock file.

  Backups and a sha256 manifest are taken first. Pre-run and post-run manifests (run3's pattern) show that no other file of the store changed.
- **Label: neutral** for scoring. The route decides where the run reads and writes; it changes no criterion, threshold or reading.
  - It is user-approved. The user chose it, option (B) of roadmap §3 U3, over the roadmap's recommendation, a read-only copy (option (A), labelled neutral).
  - The roadmap gave (B) no label and noted that it needs the owner's agreement. The user confirmed that agreement (above).

**MS1's G1 input: the QC-stage recipe** (adopted by the orchestrator on 2026-10-14, after both recipes below had been computed and before any gate M score is read).
- *Why a recipe is chosen.* MS1's own QC-stage outputs are symlinks into a home directory that cannot be read, so A1 recomputed G1 with code `515017d`.
  - Report: `annotation_dev/ms1_inplace_20261014/A1_REPORT.md` §3.
  - Files: `g1/g1_recompute.py` and `g1/g1_summary.json` in that directory, with checksums in `g1/SHA256SUMS`.
- *Adopted: the pipeline's QC-stage recipe.* `merxen.qc.registration.compute_segmentation_registration_qc` on the segmentation's shapes in the store, with the reference shapes `merscope_cell_boundaries`.
  - This is what the QC stage computes, and what the mouse gate reads in production: `mouse_gate.find_registration_check` takes the QC stage's `<sample>_registration_qc.json` (`src/merxen/annotation/mouse_gate.py` lines 187–195).
  - On MS2 it reproduces the published QC-stage JSONs exactly for all four segmentations (`g1/qc_recipe_ms2_check.json`).
  - The staged files are `g1/qc_recipe/ms1_<segmentation>_registration_qc.json`.
- *Not adopted: M6's standalone recipe* (`$A/m6/stageC/scripts/g1_registration_stageC.py`: centroids from the clustered H5AD's `obsm['spatial']`, with the original_seg H5AD as the reference). At `515017d` it reproduces M6's stage-C JSONs exactly.

| Segmentation | Density ratio, M6 recipe | Density ratio, QC-stage recipe | G1, M6 recipe | G1, QC-stage recipe |
|---|---|---|---|---|
| original_seg | 1.625 | 1.882 | warning | warning |
| proseg_hybrid | 2.197 | 1.984 | pass | warning |
| proseg_mask | 1.984 | 1.909 | warning | warning |
| reseg | 2.167 | 1.879 | pass | warning |

- *G1's lines, from the code.* G1 fails when the density ratio is below 1.5 or the shift against the platform's own segmentation exceeds 5 µm. It warns when the ratio is below 2.0.
  - Source: the `MouseGateConfig` defaults (`src/merxen/annotation/config.py` lines 852–854), applied in `mouse_gate.py` lines 845–866; plan §7.6.
  - Every shift is 0 µm under both recipes, except original_seg's, which is not measured.
- *Result.* Neither recipe fails G1 on any segmentation. The QC-stage recipe puts a G1 warning on all four segmentations; the M6 recipe puts one on two (original_seg and proseg_mask).
- **Label: neutral; POST-HOC as to the warnings.**
  - *Post-hoc:* the recipe was chosen after both sets of values had been seen.
  - *Neutral, not a tightening:* it is G1's production input. It changes no G1 fail, so it changes no gate level. No gate M criterion reads a G1 warning on MS1: MO7 asks only that MS1 is not `failed`, "with or without the warning flag" (plan §7.6). So no criterion becomes harder to pass.
  - *Not a loosening:* on every segmentation its G1 status is at least as strict as the M6 recipe's. It adds the warnings on proseg_hybrid and reseg and removes none. Its ratio is the higher one only on original_seg (1.882 against 1.625), where both recipes warn.

#### 25.3.1 The MS1 in-place run may proceed; the region-share bundle (recorded 2026-10-14, before any gate M run)

Names follow §25.1.1.

**The question** (chat, 2026-10-14 at 13:25 UTC, item 1, verbatim):

> 1. Before the [MS1] run writes into the owner's store, have you told them when it will run? The store's `zarr.json` will become owned by you, still group-writable. If you'd rather I just go ahead, say so.

**The user's answer** (chat, 2026-10-14 at 13:38 UTC, verbatim): "1. It's ok to go ahead with whatever is needed for this file. It is for a project we both work on"

The source is `USER_DECISIONS_20261014.md` section E, which quotes the exchange verbatim.

**The reading.**
- The MS1 in-place run may proceed. The data owner co-works on the project, and no further notice to them is required.
- "whatever is needed for this file" is read as the root `zarr.json` that the question names, including its change of owner. It is not read as wider write access. The writes stay those approved under U3 (above):
  - the suffixed `_mapfirst` tables;
  - the zarr root metadata entries for them;
  - the store's write-lock file.

  Backups and manifests are still taken first.
- **Label: neutral.** The route is the one the user chose under U3. The answer removes a notice step. It changes no criterion, threshold or reading.

**The region-share bundle (neutral).**
- Gate M uses `wmb_region_share` `2bf9c01c…` (revision 2, built on 2026-10-14), from the store named above.
- It replaces "`1867f978…`, or the bundle that M6b-R builds if the selected rule needs component (a)'s structure homes" in the bundle list above.
- The selected rule `a+f:tile200,nn2` (§24's amendment) reads component (a)'s structure-level subclass homes (`region_structure_home.parquet`). Only revision 2 carries them (`REGION_SHARE_REVISION` in `src/merxen/annotation/reference.py`).
- `1867f978…` (no revision field, built on 2026-10-13) is not used.
- This is the bundle the earlier text anticipated, so the label is neutral.

## 26. M13b readings (U7; recorded 2026-10-14, before any mouse gate-P mapping)

M13b (plan §12 M13b; the M13 chunks C20–C22) brings gate P to mouse:
- C20a: the second WMB test draw;
- C20b: the mouse gate-P path;
- C21: the MS1 and MS2 dry run;
- C22: the mouse real-data QC in RESOLVE.

**The gate-P text that applies (corrected 2026-10-14, in the same PR, before any mouse gate-P mapping).** The gate-P definitions and rulings of §23.9–§23.19 apply to M13b as they do to every family, **as revised by §23.21 (R1–R7) and §23.22**. That revised text is the registered gate-P text now, and the mouse dry run runs under it. **Label: neutral.**
- *Why it applies to mouse.* The user approved R1–R7 on 2026-10-07 ("Approve all six (Recommended)", §23.21) as a revision of the gate-P criteria, and §23.22 records how the code reads R3 (c).
- *The code.* The scoring code at this branch's base (`515017d`) implements the revision for every family and species. In `src/merxen/annotation/gate_p.py`, the module docstring (lines 17–55) names the function of each item: `np3_test_cell_weights` (R1), `gate_p_depth_walk` (R2), `np5_tstar_consequence` (R3 (c)), `dry_run_verdict` (R5, lines 8265–8366), `np5_decision_agreement` (R6) and `np5_rederive_ensemble` (R7). The mouse driver path does not exist yet (C20b; `GatePOptions.check_supported` refuses mouse, `gate_p_run.py` lines 316–329). It is to call the same functions.
- *Rulings of §23.19 that stand:*
  - CK2;
  - A27 (a): a mouse family's NP6 takes the human cross-platform offsets, with every gene resampled (§23.15 item 6). The label A27 is the M13 decision memo's (`$A/m13/M13_DECISION_MEMO.answered_20261007.md`, answered on 2026-10-07; §23.19).
- *CK1 (a) does not stand.* §23.21 R5: "in the dry run, a class H18 expects whose record is `not_evaluable` is reported, not failed. This replaces CK1 (a)."
- *Stricter alternative, not taken:* CK1 (a) for mouse, under which an H18 class that is not evaluable fails the mouse dry run. It is not taken because R5 replaced CK1 (a) for every family. Applying CK1 (a) to mouse alone would re-instate a ruling the user replaced.
- *The first version of this section* (commit `e54cce1`) cited CK1 (a) here and in reading (iv) and named only §23.9–§23.19. This paragraph and reading (iv) replace that text.

No mouse gate-P cell has been mapped, and no mouse gate-P output exists. No threshold, target or criterion changes here.

**The question** (put on 2026-10-13 at 16:46 UTC, verbatim): "U7, M13b (mouse gate P): start the mouse dry-run work (second WMB draw, [MS1]/[MS2] dry run) now in parallel with gate M, version 6 first, run as registered and revise under the CK2 path if it fails as predicted?"

**The user's answer** (2026-10-14 at 08:09 UTC, verbatim): "Yes, start now (Recommended)". The option text read: "Run as registered; any revision comes back to you as a loosening; the mouse NP9 time reference (OD-E18 analogue) is decided before any mouse family is scored."

**The readings** (roadmap §3 U7, with its labels):
- **(i) C20 and C21 start before M6b closes (neutral).** Only C22 needs M6b; plan §12 said "after M6b". The answer approves this ("now in parallel with gate M").
- **(ii) Draw 2's top-up (neutral; adopted by the orchestrator).** The top-up "up to 700" (§9, "Class set C_P") is read pooled with draw 1's check half.
  - Feasibility was estimated read-only before this record (roadmap §1): draw 2 can top up 26 of the 34 classes to 700, and C_P then holds about 95% of the pooled test cells.
- **(iii) Version 6 first (tightening).** A forced version 7 runs only when a mouse version-7 family is scored, on a separate version-7 test set with the v7.6 top-up. The answer approves this ("version 6 first").
- **(iv) The predicted failure, disclosed now; re-examined under R5 on 2026-10-14.**
  - Roadmap §1 M13b, verbatim (made under CK1 (a)): "**the version-6 mouse dry run is predicted to fail by construction**: [MS2]'s frozen PREP emits nothing at class for `04 DG-IMN Glut` and `29 CB Glut`, and [MS1]'s also nothing for `34 Immune`."
  - *The record such a class gets* (code at `515017d`):
    1. Version 6 freezes the PREP bundle's own decisions, in the provisional regime (`gate_p_run.py` lines 185 and 948–965).
    2. A call is confident only where the bin of its called class at that level and depth is emitted with a threshold (`resolvability.frozen_confident_mask`, `resolvability.py` lines 4922–4967). A class that emits nothing at class level therefore has no confident class-level call.
    3. `resolvability.gate_p_tested_sets` then gives it no tested set: even every bin pooled holds fewer than n_min = 200 confident calls (lines 4903–4913).
    4. `gate_p_depth_walk` marks every criterion not evaluable for a key without a tested set (`gate_p.py` lines 6056–6059). Its class-level parts are not read, so no criterion fails.
    5. `gate_p_class_records` gives every C_P class a record (line 7512). This one is **`not_evaluable`, not failed** (lines 7518–7545).
  - *What R5 changes.* `dry_run_verdict` reports a `not_evaluable` row, instead of failing it, only when the row is one H18 expects and is not a C_P class at the minimum level (`gate_p.py` lines 8302–8341).
    - For mouse the minimum level is class (`diagnostics.MIN_SIMULATION_LEVEL`, `diagnostics.py` line 182). H18's rows are at the leaf level, subclass (`h18_expected_classes`, lines 8231–8262).
    - A C_P class at class level that is not validated has `passed` False (line 8339). The class level is then incomplete in the level walk (lines 7629–7633), which fails `passes` as well (lines 8343–8350).
    - Under CK1 (a), too, a C_P class at class level had to be validated, so the roadmap's prediction did not rest on CK1 (a) at class level. R5 changes the outcome only for subclass-level H18 rows that are not evaluable: CK1 (a) failed them, and R5 reports them.
  - **The corrected prediction.** The version-6 dry run of a family is predicted to fail by construction when its C_P at class level holds one of these classes:
    - MS2: `04 DG-IMN Glut` or `29 CB Glut`;
    - MS1: either of those, or `34 Immune`.
    - For a class that C_P leaves out (fewer than 700 pooled test cells), the prediction is withdrawn. The class-level rule does not reach it, and its subclass-level H18 row, if not evaluable, is reported (R5), not failed.
    - C_P is fixed by C20 from the pooled test-cell table (draw 1's check half and draw 2) before any cell is mapped. This record does not fix it.
  - *Read while this correction was written.* These are reference metadata only: no cell was mapped and no gate-P code was run.
    - The class-level provisional decisions in `resolvability.parquet` of the two bundles (MS2 `af3fab09…`, MS1 `97a086d1…`) confirm the roadmap's premise. Neither emits any class-level bin for `04 DG-IMN Glut` or `29 CB Glut`, and MS1's emits none for `34 Immune`.
    - The test cells of these three classes: 70, 30 and 230 in draw 1 (`wmb_selfmap_testset` `c499a7ea…`, both halves); 666, 687 and 590 in each of the two draw-2 test-set bundles already in the store (`947410bd…` and `d03f52b9…`, built 2026-10-14, not yet recorded). With draw 1's check half, each class lies near C_P's 700. Whether each reaches 700 is settled when C20 fixes C_P.
  - The dry run runs as registered (§9, "Dry run"; §23.9 item 8, with D28, as revised by §23.21 R5).
  - If it fails, the criteria are revised only under CK2, with the user's approval: a revision memo, then a revision PR the user approves, then a re-run under the revised criteria that must pass. "A failed dry run is never re-scored under another reading" (§23.19).
  - Any such revision is a LOOSENING, POST-HOC at that point.
  - The answer approves this ("run as registered and revise under the CK2 path if it fails as predicted").
- **(v) NP9's time reference for mouse version-7 families stays open.**
  - OD-E18 / D10 (a) uses set a's ~0.63 ms per cell, against ~3.2 ms per cell for WMB mapping, so mouse families would fail it by design.
  - The roadmap recommends 1.5 × the mouse dry run's time per cell. That is a LOOSENING.
  - The user decides it in writing before C23, that is, before any mouse family is scored, as the answer's option text says.
- **(vi) NP5's depth source for the dry run (neutral; adopted by the orchestrator).** MS1 proseg_hybrid and MS2 original_seg, from their map_first profiles.
