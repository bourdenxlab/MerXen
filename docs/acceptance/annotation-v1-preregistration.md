# Annotation v1 pre-registration: acceptance thresholds and M3 baselines

- **Effort:** robust cell-type annotation (`robust-celltype-annotation`), milestone M3 (plan §12 "M3", items 1–7 of the shadow programme; items 2–7 are in §11).
- **Plan:** `docs/plans/robust-celltype-annotation-plan.md` rev3; the thresholds in §9 below are its §14, copied verbatim (plan file sha256 `14804ff24dcd46ab7d649775bb25aba1b3bac08d533ddd93d457a87f68895024`, last changed in `06a04a1`).
- **Measured:** 2026-09-27, branch `feature/rca-m3-mmc-engine` (MAP engine `2f346b9`, `96f2834`; shadow metrics `b4a71fe`; baseline script `ed373e5`; items 2–7: LL (vii) `89266ff`, shadow metrics `0afc4d6`, scripts `a6b3839`). **Re-measured after the M3 review** with `2a3f183`, `92d9cd9` and `29d1660`: one tile grid resampled jointly for both sections (H1 CIs), E2's SEA-AD probability definitions, a float32 tolerance on every threshold test, the H4 COP-rule option and both OD-B8 readings. The launcher log of every rerun records the git commit and the script's sha256 (`$A/shadow/<item>/PROGRESS.txt`). The pre-review outputs are in `$A/shadow/superseded_20260927_pre_review/`.
- **Evidence root** (`$A`): `/srv/storage/MerXen/annotation_dev/evidence_20260926`. The item-1 baselines are in `$A/shadow/baselines/metrics/*.csv` (§10 lists the files); items 2–7 are in `$A/shadow/{e8,x1,heldout,flags,ll,glial_jsd}/` (§11), summarised in `$A/shadow/SHADOW_SUMMARY.txt`.

## 1. Rules

1. **The thresholds are fixed by this document.** They are the plan's §14 values, copied verbatim in §9. No threshold is changed here.
2. **Baselines may only tighten a threshold.** A measured baseline can justify a stricter threshold, decided in the M3 PR. Loosening a threshold needs the user's written approval in the gate PR. That includes redefining a metric so that it becomes easier to pass, or dropping a dataset, a class or a criterion. This applies to gate H (M8), gate M (M9) and gate P.
3. **Held-out validation sets are reported separately and are not used for tuning:** the donors P7113 and P5011 (§4) and the second segmentation, reseg (§5; H17 is scored on P7513 and P1212 reseg; P7113 and P5011 reseg are reported for information). The development datasets are P7513 and P1212 on proseg_hybrid (§3). Held-out results never select a rule variant, threshold or floor. The pre-registered selection rules (MO10's variant at M6b, gate P's evaluation rules) are the only exceptions. **Nothing in M3 was tuned on the donors, but E2 was:** the raw thresholds, the packaged floors, the SEA-AD subclass thresholds and the donors' own H1 and H7 thresholds were derived with P7113 and P5011 data (§4, first table). H1 and H7 on the donors therefore test consistency with their own E2 measurement, not out-of-sample performance.
4. **Metric definitions are the ones in §2.** RESOLVE (M4) and the acceptance scripts (M8) must compute the gate metrics the same way or document the difference in the gate PR. A difference that makes a criterion easier to pass counts as a loosening (rule 2).

## 2. What was measured, and how

**Inputs (read-only).** These are the published `<pair>_<PLAT>_clustered.h5ad` of P7513, P1212, P7113 and P5011 × MERSCOPE / Xenium, on proseg_hybrid and reseg: `/srv/storage/MerXen/results/<pair>/<seg>/clustering_squidpy/clustering_squidpy_out/<plat>/`. The runs use `layers["counts"]` on the table cells (every object in the clustered file; `total_counts >= 10` after control removal). The shared tissue mask comes from `results/<pair>/alignment/align_out/shared_tissue_mask.npy`, in the Xenium frame of `registration_summary.json`. The MERSCOPE coordinates are already in that frame (shape key `*_aligned_nonrigid`). The segmented-object counts come from `$A/research/lowcount/qc_summary.csv`.

**MAP runs (M3 stage A code).** Each pair × segmentation ran `merxen annotate --species human --from-clustered-h5ad <M> --from-clustered-h5ad <X> --store /media/mathieubo/SSD1/MerXen/annotation_references --gene-id-fallback-csv <WHB-10Xv3-Nonneurons-raw.h5ad> --n-processors 6 --out $A/shadow/baselines/runs/<pair>/<seg>`, with at most three jobs at once. The engine settings were bootstrap factor 0.5, 100 iterations, seed 0, 6 processes, OMP / OpenBLAS / MKL / numba threads 1, raw normalisation, and cell_type_mapper 1.7.2 (`824caef`).
- The bundles were builder v2: WHB frontal set a `b70dc181` (panel `6e5fd5fb`, 297 genes, post-M0e), WHB frontal set c `f20d11b0` (panel `77bd80ed`, 265 genes, curated E5 family; proseg_hybrid only) and SEA-AD Multiregion `3973770c` (set a).
- **Declared panel for reseg.** The published reseg MERSCOPE `var` of P1212 (299 features) and P5011 (268) is `min_cells`-filtered. A panel derived from it would get a new hash that no bundle has. Plan §3.2 requires the declared panel instead. Those two pairs were therefore re-run through `merxen annotation-panel --panel-file <PLAT>=<same section's proseg_hybrid clustered H5AD>`: that file's unfiltered 300-feature `var` is the declared panel. The re-run gave set a `6e5fd5fb`, and `merxen annotate --panel-dir` then mapped the genes present. P1212 reseg MERSCOPE lacks 1 of the 297 genes. P5011 reseg MERSCOPE lacks 32 of 297 (10.8%). Both were mapped with a restricted lookup (M3 behaviour; §7 item 4).

**Shadow v1 rules** (`merxen.annotation.shadow.evaluate_human_rules`). RESOLVE is M4, so the confident statuses come from a shadow evaluation of plan §5.2 rules 1, 2 and 4 and the §5.4 gate. It uses the raw v1 thresholds (WHB broad / lineage 0.73, supercluster 0.69, SEA-AD broad 0.68) and the packaged set-a floors (`floors_human.csv`). Probabilities are stored as float32 (tidy parquet, `ct_*_raw`), so every threshold test allows 1e-6 below the threshold (`schema.meets_threshold`); without it a bootstrap probability of exactly 0.69 fails 0.69 (0.7–1.3% of table cells sit exactly at the supercluster threshold). There are three rules:
- **Lineage** needs aggregated bp >= 0.73, a plausible node (an implausible node keeps its lineage when SEA-AD agrees at lineage) and the second vote.
- **Broad** needs a confident lineage, aggregated bp >= 0.73, counts >= the class × platform broad floor, and the SEA-AD rule. Below 60 counts SEA-AD must agree at the 7-class level. From 60 counts SEA-AD must not confidently disagree. The COP rule applies: a COP call is broad OPC only with >= 120 counts and supercluster bp >= 0.69, or with a confident SEA-AD OPC call.
- **Supercluster** needs a confident broad, gate level `full`, bp >= 0.69 and counts >= the supercluster floor.

The rules have no resolvability (M3b) and no flags. M4's exit requires RESOLVE's coverage to be within ±0.05 of these baselines.

**Metric definitions.**

| Criterion | Operational definition (code) |
|---|---|
| H1 | Soft broad composition: per table cell, the bootstrap probability of the assigned WHB supercluster and of its 5 runner-ups, aggregated to the 7 broad classes through `whb_supercluster_vocab.csv`. Sinks and nodes outside the 7 classes, plus the residual 1 − Σ, are "unallocated" (`soft_matrix_from_provisional`). The composition is the sum over cells. The JSD is the base-2 Jensen-Shannon **distance** (as `scipy.spatial.distance.jensenshannon` and E1 `real_jsd.csv`) of the renormalised 7-class vectors, MERSCOPE vs Xenium. The 95% CI is a spatial block bootstrap over **matched tile locations**: both sections are tiled on one grid of 500 µm squares in the shared Xenium frame (the MERSCOPE `*_aligned_nonrigid` coordinates; inside `shared_tissue_mask` for the mask variant), and each of 200 replicates draws the tile locations (non-empty in either section) with replacement and applies the same multinomial weights to both sections (seed 0, percentile interval; `shared_tile_codes`, `tile_sums`, `block_bootstrap_jsd(resampling="joint")`). Paired differences of two labellings reuse the same draw for both labellings (`paired_block_bootstrap_jsd_difference`). The interval therefore reflects which tissue is sampled, not anatomy mismatched between the adjacent sections. Resampling each section's tiles independently is reported as a sensitivity only (`ci_*_independent` in `jsd.csv`; it was the pre-review method and widens the P7513 interval from [0.116, 0.143] to [0.095, 0.166]). It is reported on the whole section and inside the shared tissue mask. Also reported: argmax (broad class of the assigned supercluster), confident-only (the v1 shadow confident broad label), the soft composition over cells with >= 30 counts, and set c. |
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
| M3 exit | Production vs the E1 (iii) pilot (`$A/research/pilot/{whb_region,seaad_mr}`, P7513 and P1212 30k subsets). The comparison covers the WHB supercluster assignment on cells with production bp >= 0.8 (target >= 0.98) and the SEA-AD subclass assignment on all shared cells (target >= 0.945 − 0.02 = 0.925). P7113 and P5011 are compared with E2's runs of the pilot configuration (`$A/exp/E2/mmc/{WHBF,SEAAD}__<sid>_sub30k.json`), for information. |

**Code.** Metrics: `src/merxen/annotation/shadow.py` (tests: `tests/test_annotation/test_shadow.py`). Driver: `scripts/acceptance/shadow_baselines.py --runs-root $A/shadow/baselines/runs --results-root /srv/storage/MerXen/results --evidence-root $A --out $A/shadow/baselines/metrics` (86 s, 3.5 GB; the shadow scripts take the gene-ID fallback tables from `--gene-id-fallback-csv` or `$MERXEN_GENE_ID_FALLBACK_CSV`).

## 3. Development baselines: P7513 and P1212, proseg_hybrid

These are the rows that the §9 human flip rule scores for H1–H3, H5, H7–H10 and H12–H16.

| Pair | H1 threshold | Soft, whole section [95% CI] | Soft, shared mask [95% CI] | Argmax, whole | Confident, whole | Soft, cells >= 30 counts | Set c soft / argmax | Unallocated soft mass M / X |
|---|---|---|---|---|---|---|---|---|
| P7513 | <= 0.17 | 0.129 [0.116, 0.143] | 0.128 [0.121, 0.134] | 0.130 [0.118, 0.143] | 0.145 [0.130, 0.163] | 0.211 | 0.121 / 0.121 | 0.017 / 0.017 |
| P1212 | <= 0.17 | 0.133 [0.126, 0.142] | 0.154 [0.148, 0.160] | 0.149 [0.138, 0.161] | 0.206 [0.196, 0.220] | 0.183 | 0.109 / 0.117 | 0.019 / 0.020 |

| Sample | Table cells | Median counts | A (share >= 30 counts) | Gate level | Warning | Confident broad, table cells [H7 threshold] | Confident broad, segmented objects | Lineage | Supercluster | Implausible calls (H2 <= 1%) | WHB-SEA 7-class, >= 20 counts [H3 threshold] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P7513_MERSCOPE | 164,370 | 71 | 0.759 | full | no | 0.704 [>= 0.62] | 0.546 | 0.745 | 0.620 | 0.60% | 0.926 [>= 0.90] |
| P7513_XENIUM | 132,489 | 43 | 0.641 | full | no | 0.597 [>= 0.44] | 0.471 | 0.640 | 0.519 | 0.35% | 0.917 [>= 0.90] |
| P1212_MERSCOPE | 102,886 | 17 | 0.234 | broad_only | no | 0.438 [>= 0.34] | 0.152 | 0.598 | 0.000 | 0.30% | 0.825 [>= 0.80] |
| P1212_XENIUM | 157,676 | 25 | 0.426 | full | no | 0.510 [>= 0.29] | 0.332 | 0.589 | 0.446 | 0.36% | 0.908 [>= 0.80] |

| Sample | COP argmax calls | Confident COP supercluster (H5 <= 2%) | Confident broad OPC (H5 <= 10%) | COP-derived share of confident OPC | COP suppressed (`flag_cop_suppressed`) | New vs legacy referee: disputes (share); new / legacy / tie (H10 >= 0.70) | WHB vs SEA-AD referee: disputes (share); WHB / SEA-AD / tie |
|---|---|---|---|---|---|---|---|
| P7513_MERSCOPE | 4.14% | 0.28% | 2.32% | 0.620 | 0.86% | 50,969 (0.310); 0.989 / 0.005 / 0.006 | 17,683 (0.108); 0.395 / 0.339 / 0.267 |
| P7513_XENIUM | 2.80% | 0.12% | 1.51% | 0.623 | 0.54% | 20,677 (0.156); 0.980 / 0.007 / 0.012 | 13,667 (0.103); 0.466 / 0.370 / 0.164 |
| P1212_MERSCOPE | 8.90% | 0.00% | 1.43% | 0.823 | 3.50% | 43,548 (0.423); 0.961 / 0.006 / 0.032 | 22,384 (0.218); 0.298 / 0.314 / 0.388 |
| P1212_XENIUM | 4.18% | 0.02% | 1.14% | 0.691 | 1.08% | 24,576 (0.156); 0.978 / 0.008 / 0.014 | 19,822 (0.126); 0.369 / 0.365 / 0.266 |

- **H1.** The soft JSD is 0.129 (P7513) and 0.133 (P1212), against a threshold of 0.17. The plan's basis (E2 `cells_native`) was 0.139 / 0.129. The argmax JSD is 0.130 / 0.149, against E1 `real_jsd.csv` (iii) 0.132 / 0.143. With matched tiles the intervals are 0.116–0.143 and 0.126–0.142. The pre-review intervals (0.095–0.166 and 0.122–0.154) resampled each section's tiles independently, which mismatches anatomy between the adjacent sections (extra white matter on one side, extra grey matter on the other); their width was that artefact, not tile heterogeneity (§2). Confident-only compositions are further apart (0.145 / 0.206; the draft floors gave 0.236 / 0.285), which is why H1 is soft (D-C5).
- **H2.** The implausible share is 0.30–0.60%, within the 1% threshold (E1 (iii): 0.3–0.6%). The calls go to Amygdala excitatory (region-implausible), Miscellaneous and Splatter (`implausible_top_nodes`).
- **H3.** Agreement is 0.926 / 0.917 / 0.825 / 0.908 (E1: 0.932 / 0.917 / 0.826 / 0.904).
- **H5.** Confident COP supercluster calls are at most 0.28% and confident broad OPC calls at most 2.3%. COP-derived calls make up 62–82% of confident OPC (plan: 62–86%).
- **H7.** Coverage clears its threshold by 0.08 to 0.22.
- **H8.** P1212_MERSCOPE is `broad_only` (A = 0.234). Its coverage of segmented objects, 0.152, is only 0.002 above the warning threshold (§7 item 3). Every other development sample is `full`.
- **H10.** Markers side with the new label in 96–99% of new-vs-legacy disputes. In WHB-vs-SEA-AD disputes the referee is close to a tie (E1: "roughly a tie").
- **OD-B13.** The below-60 SEA-AD rule costs 0.3–0.9 points of coverage. Replacing it with the likelihood typer would cost 2.0–5.8 points, so the LL variant keeps less coverage than the SEA-AD rule, not more. The trigger is not met.

Second vote on E2's 30k native cells (coverage of those cells; `second_vote_cost.csv`):

| Sample | Shared cells (share < 60 counts) | No second method | v1 without the below-60 rule | v1 (SEA-AD below 60) | v1 with LL below 60 | E2 LL rule alone | Below-60 SEA-AD cost | Below-60 LL cost | LL minus SEA-AD (trigger if > +0.08) |
|---|---|---|---|---|---|---|---|---|---|
| P7513_MERSCOPE | 30,000 (0.44) | 0.698 | 0.710 | 0.704 | 0.690 | 0.679 | 0.006 | 0.020 | -0.014 |
| P7513_XENIUM | 30,000 (0.61) | 0.597 | 0.606 | 0.603 | 0.551 | 0.543 | 0.003 | 0.054 | -0.052 |
| P1212_MERSCOPE | 30,000 (0.92) | 0.436 | 0.447 | 0.438 | 0.394 | 0.383 | 0.009 | 0.053 | -0.044 |
| P1212_XENIUM | 30,000 (0.83) | 0.500 | 0.508 | 0.505 | 0.451 | 0.443 | 0.003 | 0.058 | -0.055 |

M3 exit check against the E1 (iii) pilot (`exit_checks.csv`):

| Sample | Comparator | WHB supercluster, production bp >= 0.8 (n) [>= 0.98] | WHB supercluster, all shared cells | SEA-AD subclass, all shared cells [>= 0.925] | SEA-AD subclass, production bp >= 0.8 |
|---|---|---|---|---|---|
| P7513_MERSCOPE | E1 (iii) pilot | 1.0000 (16,644 of 30,000) | 0.9587 | 0.9573 | 0.9988 |
| P7513_XENIUM | E1 (iii) pilot | 1.0000 (13,625 of 30,000) | 0.9519 | 0.9453 | 0.9973 |
| P1212_MERSCOPE | E1 (iii) pilot | 1.0000 (9,722 of 30,000) | 0.9272 | 0.9483 | 0.9991 |
| P1212_XENIUM | E1 (iii) pilot | 1.0000 (11,275 of 30,000) | 0.9428 | 0.9413 | 0.9983 |

Both exit criteria hold on all four samples: the WHB supercluster agreement is 1.0000 at production bp >= 0.8, and the SEA-AD subclass agreement is 0.941–0.957 against 0.925. The third exit criterion, wall time within §10, is in §6.

## 4. Held-out donors: P7113 and P5011, proseg_hybrid (reported separately)

The flip rule scores H1–H3, H7, H8 and H12 on these datasets. Nothing in M3 was tuned on them, but E2, the evidence behind v1, used their data for these parameters and thresholds:

| Parameter or threshold | Derived with P7113 / P5011 data | Source |
|---|---|---|
| Raw thresholds (WHB broad / lineage 0.73, supercluster 0.69, SEA-AD broad 0.68) | Thinned high-count cells of P7113 M, P7113 X and P5011 X (`CALIB`, pooled with P7513 M / X and P1212 X) | `exp/E2/thresholds2.py` (`R2_pool = calib_real(T.ds.isin(CALIB))`), `exp/E2/common.py` |
| SEA-AD subclass thresholds (0.55 below 60 counts, 0.45 from 60) | The same `CALIB` pool | `exp/E2/thresholds2.py` R2 pooled |
| Packaged floors (`floors_human.csv`, broad and supercluster per class × platform) | The same `CALIB` datasets | `exp/E2/floors.py` (`T.ds.isin(CALIB)`) |
| H1 thresholds P7113 ≤ 0.17, P5011 ≤ 0.31 | Each donor's own E2 soft JSD (0.136 / 0.279) + 0.03 | Plan §14 H1 basis; E2 `cells_native` |
| H7 thresholds P7113 M ≥ 0.67, X ≥ 0.39; P5011 M ≥ 0.30, X ≥ 0.40 | Each dataset's own E2 LL-rule coverage − 0.10 | Plan §14 H7 basis; `exp/E2/out/final_coverage_and_gate.csv` |

So H1 and H7 on P7113 and P5011 test whether production reproduces the donors' own E2 measurement, not out-of-sample performance, and every rule on them runs with thresholds and floors partly fitted to their cells. What remains genuinely held out: the reseg segmentation (H17, §5; never used by E2) and, on the donors, H2, H3, H8 and H12, whose thresholds are generic (H2 1%, H3 0.80, H8 mechanical, H12 label-independent anatomy) rather than derived from the donors' values.

| Pair | H1 threshold | Soft, whole section [95% CI] | Soft, shared mask [95% CI] | Argmax, whole | Confident, whole | Soft, cells >= 30 counts | Set c soft / argmax | Unallocated soft mass M / X |
|---|---|---|---|---|---|---|---|---|
| P7113 | <= 0.17 | 0.126 [0.112, 0.143] | 0.107 [0.099, 0.117] | 0.110 [0.096, 0.126] | 0.156 [0.142, 0.167] | 0.326 | 0.110 / 0.096 | 0.016 / 0.017 |
| P5011 | <= 0.31 | 0.227 [0.212, 0.242] | 0.235 [0.218, 0.250] | 0.253 [0.237, 0.269] | 0.324 [0.307, 0.340] | 0.243 | 0.161 / 0.183 | 0.038 / 0.030 |

| Sample | Table cells | Median counts | A (share >= 30 counts) | Gate level | Warning | Confident broad, table cells [H7 threshold] | Confident broad, segmented objects | Lineage | Supercluster | Implausible calls (H2 <= 1%) | WHB-SEA 7-class, >= 20 counts [H3 threshold] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P7113_MERSCOPE | 96,168 | 46 | 0.667 | full | no | 0.756 [>= 0.67] | 0.371 | 0.795 | 0.678 | 0.77% | 0.948 [>= 0.80] |
| P7113_XENIUM | 163,381 | 32 | 0.541 | full | no | 0.648 [>= 0.39] | 0.496 | 0.708 | 0.564 | 0.59% | 0.929 [>= 0.80] |
| P5011_MERSCOPE | 58,565 | 16 | 0.129 | broad_only | yes | 0.459 [>= 0.30] | 0.105 | 0.558 | 0.000 | 1.02% | 0.828 [>= 0.80] |
| P5011_XENIUM | 114,294 | 28 | 0.472 | full | no | 0.556 [>= 0.40] | 0.318 | 0.634 | 0.373 | 0.91% | 0.921 [>= 0.80] |

| Sample | COP argmax calls | Confident COP supercluster (H5 <= 2%) | Confident broad OPC (H5 <= 10%) | COP-derived share of confident OPC | COP suppressed (`flag_cop_suppressed`) | New vs legacy referee: disputes (share); new / legacy / tie (H10 >= 0.70) | WHB vs SEA-AD referee: disputes (share); WHB / SEA-AD / tie |
|---|---|---|---|---|---|---|---|
| P7113_MERSCOPE | 3.28% | 0.13% | 2.24% | 0.679 | 0.46% | 42,102 (0.438); 0.991 / 0.004 / 0.005 | 8,085 (0.084); 0.327 / 0.345 / 0.328 |
| P7113_XENIUM | 3.51% | 0.10% | 1.91% | 0.638 | 0.71% | 13,448 (0.082); 0.897 / 0.044 / 0.059 | 14,908 (0.091); 0.410 / 0.376 / 0.214 |
| P5011_MERSCOPE | 8.75% | 0.00% | 2.92% | 0.722 | 1.70% | 11,565 (0.197); 0.940 / 0.016 / 0.044 | 13,004 (0.222); 0.315 / 0.324 / 0.361 |
| P5011_XENIUM | 3.73% | 0.03% | 2.06% | 0.620 | 0.58% | 10,249 (0.090); 0.926 / 0.038 / 0.035 | 12,509 (0.109); 0.406 / 0.360 / 0.234 |

- **H1.** P7113 is 0.126 (CI 0.112–0.143; threshold 0.17) and P5011 is 0.227 (0.212–0.242; threshold 0.31). The plan's basis was 0.136 / 0.279 (the donors' own E2 values, table above). The pre-review independent-resampling upper bound for P7113 (0.171) crossed 0.17 purely as an artefact of mismatched anatomy.
- **H2.** P7113 is at 0.77% / 0.59%. **P5011_MERSCOPE is at 1.02%, above the 1% threshold** (§7 item 1); P5011_XENIUM is at 0.91%.
- **H3.** Agreement is 0.948 / 0.929 / 0.828 / 0.921, all at or above 0.80.
- **H7.** Every sample clears its threshold, by 0.09 to 0.26.
- **H8.** P5011_MERSCOPE is `broad_only` (A = 0.129) with the warning flag (coverage of segmented objects 0.105), as §5.4 expects. P7113 and P5011_XENIUM are `full`.
- **OD-B13.** The trigger is not met.

| Sample | Shared cells (share < 60 counts) | No second method | v1 without the below-60 rule | v1 (SEA-AD below 60) | v1 with LL below 60 | E2 LL rule alone | Below-60 SEA-AD cost | Below-60 LL cost | LL minus SEA-AD (trigger if > +0.08) |
|---|---|---|---|---|---|---|---|---|---|
| P7113_MERSCOPE | 30,000 (0.61) | 0.747 | 0.762 | 0.757 | 0.742 | 0.727 | 0.005 | 0.020 | -0.015 |
| P7113_XENIUM | 30,000 (0.71) | 0.637 | 0.648 | 0.646 | 0.572 | 0.561 | 0.002 | 0.076 | -0.073 |
| P5011_MERSCOPE | 30,000 (0.99) | 0.442 | 0.464 | 0.454 | 0.427 | 0.406 | 0.010 | 0.037 | -0.027 |
| P5011_XENIUM | 30,000 (0.77) | 0.543 | 0.556 | 0.551 | 0.535 | 0.523 | 0.004 | 0.021 | -0.016 |

The same MAP compared with E2's runs of the pilot configuration (for information; the formal exit check is §3):

| Sample | Comparator | WHB supercluster, production bp >= 0.8 (n) [>= 0.98] | WHB supercluster, all shared cells | SEA-AD subclass, all shared cells [>= 0.925] | SEA-AD subclass, production bp >= 0.8 |
|---|---|---|---|---|---|
| P7113_MERSCOPE | E2 run of the pilot configuration | 1.0000 (19,061 of 30,000) | 0.9701 | 0.9739 | 0.9995 |
| P7113_XENIUM | E2 run of the pilot configuration | 1.0000 (15,371 of 30,000) | 0.9573 | 0.9563 | 0.9988 |
| P5011_MERSCOPE | E2 run of the pilot configuration | 1.0000 (8,163 of 30,000) | 0.9987 | 0.9986 | 1.0000 |
| P5011_XENIUM | E2 run of the pilot configuration | 1.0000 (9,960 of 30,000) | 0.9397 | 0.9295 | 0.9968 |

## 5. Held-out segmentation: reseg (reported separately)

H17 is scored on P7513 and P1212 reseg: the H1 thresholds + 0.02, H2 and H9. The table shows H1 + 0.02 for every pair. P7113 and P5011 reseg are for information only. Reseg has no set-c run (the default `annotation_xplat_sensitivity_segmentations` is proseg_hybrid). P1212 and P5011 reseg used the declared panel (§2), and P5011 reseg MERSCOPE lacks 32 of the 297 panel genes (§7 item 4).

| Pair | H1 threshold | Soft, whole section [95% CI] | Soft, shared mask [95% CI] | Argmax, whole | Confident, whole | Soft, cells >= 30 counts | Set c soft / argmax | Unallocated soft mass M / X |
|---|---|---|---|---|---|---|---|---|
| P7513 | <= 0.19 | 0.127 [0.114, 0.142] | 0.125 [0.118, 0.132] | 0.129 [0.115, 0.143] | 0.128 [0.116, 0.143] | 0.259 | – | 0.008 / 0.009 |
| P1212 | <= 0.19 | 0.125 [0.110, 0.146] | 0.163 [0.152, 0.179] | 0.140 [0.113, 0.173] | 0.199 [0.182, 0.223] | 0.312 | – | 0.010 / 0.012 |
| P7113 | <= 0.19 | 0.177 [0.163, 0.191] | 0.160 [0.151, 0.170] | 0.177 [0.163, 0.192] | 0.188 [0.175, 0.203] | 0.444 | – | 0.007 / 0.011 |
| P5011 | <= 0.33 | 0.320 [0.303, 0.342] | 0.323 [0.301, 0.345] | 0.337 [0.319, 0.360] | 0.342 [0.324, 0.363] | 0.294 | – | 0.033 / 0.017 |

| Sample | Table cells | Median counts | A (share >= 30 counts) | Gate level | Warning | Confident broad, table cells [H7 threshold] | Confident broad, segmented objects | Lineage | Supercluster | Implausible calls (H2 <= 1%) | WHB-SEA 7-class, >= 20 counts [H3 threshold] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P7513_MERSCOPE | 137,129 | 71 | 0.784 | full | no | 0.887 | 0.574 | 0.905 | 0.773 | 0.71% | 0.976 |
| P7513_XENIUM | 97,363 | 41 | 0.626 | full | no | 0.836 | 0.485 | 0.874 | 0.705 | 0.45% | 0.973 |
| P1212_MERSCOPE | 35,894 | 17 | 0.256 | broad_only | yes | 0.718 | 0.087 | 0.763 | 0.000 | 1.04% | 0.874 |
| P1212_XENIUM | 85,349 | 23 | 0.376 | full | no | 0.778 | 0.274 | 0.857 | 0.645 | 0.77% | 0.962 |
| P7113_MERSCOPE | 80,827 | 39 | 0.618 | full | no | 0.883 | 0.364 | 0.902 | 0.794 | 1.13% | 0.980 |
| P7113_XENIUM | 117,078 | 30 | 0.512 | full | no | 0.839 | 0.460 | 0.889 | 0.710 | 0.88% | 0.971 |
| P5011_MERSCOPE | 11,845 | 13 | 0.039 | broad_only | yes | 0.850 | 0.039 | 0.909 | 0.000 | 2.44% | 0.953 |
| P5011_XENIUM | 63,037 | 25 | 0.423 | full | no | 0.792 | 0.249 | 0.884 | 0.504 | 1.35% | 0.981 |

| Sample | COP argmax calls | Confident COP supercluster (H5 <= 2%) | Confident broad OPC (H5 <= 10%) | COP-derived share of confident OPC | COP suppressed (`flag_cop_suppressed`) | New vs legacy referee: disputes (share); new / legacy / tie (H10 >= 0.70) | WHB vs SEA-AD referee: disputes (share); WHB / SEA-AD / tie |
|---|---|---|---|---|---|---|---|
| P7513_MERSCOPE | 1.74% | 0.23% | 2.68% | 0.510 | 0.18% | 51,224 (0.374); 0.993 / 0.003 / 0.004 | 4,043 (0.029); 0.312 / 0.450 / 0.238 |
| P7513_XENIUM | 1.79% | 0.11% | 1.81% | 0.657 | 0.24% | 22,078 (0.227); 0.985 / 0.007 / 0.008 | 3,297 (0.034); 0.314 / 0.420 / 0.266 |
| P1212_MERSCOPE | 2.35% | 0.00% | 2.65% | 0.709 | 0.28% | 13,568 (0.378); 0.973 / 0.002 / 0.025 | 3,714 (0.103); 0.110 / 0.338 / 0.552 |
| P1212_XENIUM | 1.47% | 0.01% | 1.21% | 0.518 | 0.37% | 8,655 (0.101); 0.921 / 0.022 / 0.057 | 3,478 (0.041); 0.268 / 0.434 / 0.298 |
| P7113_MERSCOPE | 1.55% | 0.07% | 2.59% | 0.522 | 0.06% | 41,797 (0.517); 0.996 / 0.001 / 0.003 | 2,825 (0.035); 0.138 / 0.482 / 0.379 |
| P7113_XENIUM | 1.77% | 0.07% | 2.63% | 0.537 | 0.09% | 11,723 (0.100); 0.918 / 0.021 / 0.061 | 3,595 (0.031); 0.328 / 0.393 / 0.279 |
| P5011_MERSCOPE | 6.48% | 0.00% | 9.06% | 0.643 | 0.20% | 3,454 (0.292); 0.992 / 0.000 / 0.008 | 367 (0.031); 0.117 / 0.428 / 0.455 |
| P5011_XENIUM | 2.46% | 0.01% | 3.89% | 0.570 | 0.09% | 7,287 (0.116); 0.982 / 0.005 / 0.013 | 1,749 (0.028); 0.268 / 0.535 / 0.197 |

- **H17 / H1 + 0.02.** P7513 is 0.127 and P1212 is 0.125, both against 0.19. For information, P7113 is 0.177 (0.19 would apply) and P5011 is 0.320 (0.33).
- **H17 / H2.** P7513 is at 0.71% / 0.45%. **P1212_MERSCOPE reseg is at 1.04%, above 1%** (§7 item 1); P1212_XENIUM is at 0.77%. For information, P7113 MERSCOPE is at 1.13% and P5011 at 2.44% / 1.35%.
- **H17 / H9.** Not measured here (§8).
- Reseg has higher coverage and WHB–SEA agreement than proseg_hybrid on every sample (e.g. P7513_MERSCOPE broad coverage 0.887 vs 0.704, agreement 0.976 vs 0.926) over fewer table cells. This is input for E8 (OD-B6, OD-B7), not a criterion.

## 6. Resources (H14; §10)

| Pair | Segmentation | MAP wall (s) | MMC runs | WHB processor-s per 1k cells | SEA-AD processor-s per 1k cells | Max mapper peak RSS (GB) | Panel genes missing |
|---|---|---|---|---|---|---|---|
| P7513 | proseg_hybrid | 315 | 6 | 1.29–1.75 | 2.21–2.41 | 2.81 | 0 |
| P1212 | proseg_hybrid | 283 | 6 | 1.35–1.49 | 2.43–2.59 | 2.81 | 0 |
| P7113 | proseg_hybrid | 279 | 6 | 1.38–1.51 | 2.44–2.45 | 2.81 | 0 |
| P5011 | proseg_hybrid | 200 | 6 | 1.47–1.69 | 2.44–3.06 | 2.81 | 0 |
| P7513 | reseg | 179 | 4 | 1.32–1.49 | 2.26–2.54 | 2.81 | 0 |
| P1212 | reseg | 107 | 4 | 1.52–1.77 | 2.59–3.58 | 2.81 | MERSCOPE 1 |
| P7113 | reseg | 159 | 4 | 1.32–1.78 | 2.20–2.80 | 2.81 | 0 |
| P5011 | reseg | 80 | 4 | 1.59–3.72 | 2.82–7.50 | 2.81 | MERSCOPE 32 |

- MAP took 200–315 s per pair on proseg_hybrid, for 2 platforms × (WHB set a, WHB set c, SEA-AD). §10 budgets 25–40 min. Reseg (no set c) took 80–179 s. Both segmentations of a pair took at most 8.2 min, against H14's 2.5 h for all four segmentations.
- The mapper's peak RSS was at most 2.81 GB. The `merxen annotate` process peaked at 2.95 GB (`/usr/bin/time`, `$A/shadow/baselines/PROGRESS.txt`), against H14's 16 GB.
- WHB used 1.3–1.8 processor-seconds per 1k cells. §10 assumed 13.2–13.6; the host load was about 3–13. The 3.7 / 7.5 figures on P5011 reseg MERSCOPE (11,845 cells) are fixed start-up cost.
- No GPU was used.

## 7. Findings for the M3 PR and the gate PRs

1. **H2 fails at baseline on two scored datasets.** P5011_MERSCOPE proseg_hybrid is at 1.02% (held-out donor; the flip rule scores H2 on P5011), and P1212_MERSCOPE reseg is at 1.04% (H17 includes H2). For information, P7113_MERSCOPE reseg is at 1.13% and P5011 reseg at 2.44% / 1.35%. The calls are mostly Amygdala excitatory (region-implausible for frontal cortex in the vocab) and Miscellaneous.
   - Counting only implausible calls whose lineage probability passes 0.73 would give 0.57% and 0.93%. That changes the metric definition in the passing direction, so it is a loosening and needs the user's written approval (§1 rule 2).
   - Nothing is changed here. The gate PR needs a fix, for example in RESOLVE's status precedence (M4) or the vocab's plausibility of Amygdala excitatory, or a written, user-approved exception.
2. **OD-B13 is not triggered.** On all eight datasets the below-60 SEA-AD rule removes 0.2–1.0 points of confident broad coverage (v1 without the rule minus v1). Using the likelihood typer as the below-60 vote instead would remove 1.4–7.3 points more (`ll_minus_sea` −0.014 to −0.073). E2's LL rule alone, with no SEA-AD role, keeps 2.5–8.5 points less than v1. The LL part of M10 therefore stays in v1.1, to be recorded in the M3 PR. The LL value test (M3 item 6; the LL (vii) recipe on the WHB-frontal bundle profiles) confirms this on every table cell (§11.5).
3. **H8 margin.** P1212_MERSCOPE's coverage of segmented objects (0.152) is 0.002 above the 0.15 warning threshold, and §5.4 expects no warning there. RESOLVE's coverage may move by up to ±0.05 (M4 exit), so P1212_MERSCOPE may acquire the warning flag. The warning never changes the gate level.
4. **P5011 reseg MERSCOPE lacks 10.8% of the panel.** The published `var` of 11,845 cells is `min_cells`-filtered to 268 features. M3 maps the present genes with a restricted lookup; plan §3.3 / §8.1 route more than 5% missing genes to a panel family of its own with a full PREP (M3b). Its reseg baselines are provisional until M3b re-measures them. P1212 reseg MERSCOPE lacks 1 gene (0.34%), within the 1% that a restricted lookup covers.
5. **Standalone runs on `min_cells`-filtered outputs.** `merxen annotate --from-clustered-h5ad` derives the panel from the published `var`. When a small sample lost a gene, the derived panel gets a new hash and the store has no bundle for it (P1212 and P5011 reseg failed this way). The declared panel has to be supplied through `merxen annotation-panel --panel-file` and `--panel-dir`. Adding `--panel-file` to `merxen annotate` is an open follow-up.
6. **Candidate tightenings** (not applied; for decision in the M3 PR):
   - **H1** (basis "soft + 0.03", now measured on production): P7513 ≤ 0.16, P7113 ≤ 0.16, P5011 ≤ 0.26 (P1212 stays at 0.17).
   - **H3:** P7113 ≥ 0.90 (baseline 0.948 / 0.929).
   - **H5:** confident COP supercluster ≤ 1% (maximum baseline 0.28%) and confident broad OPC ≤ 5% (2.9%).
   - **H7** (basis "E2 LL-rule coverage − 0.10"; now the SEA-rule baseline − 0.10 where that is stricter): P7513 X ≥ 0.50, P1212 X ≥ 0.41, P7113 X ≥ 0.55, P5011 M ≥ 0.36, P5011 X ≥ 0.46.
   - **H10:** ≥ 0.85 (minimum baseline on the flip-rule datasets 0.961).

   Resolvability (M3b) and RESOLVE (M4) can still lower coverage, so tightening H7 before M4 risks a failure that belongs to M3b.
7. **H4's "assigned class" is an open choice for the user** (§11.3). Under the M3 working metric (a), the re-map's argmax over all table cells, H4 fails at baseline on P1212 (2/7 and 5/7 on MERSCOPE / Xenium) and P7113_MERSCOPE (5/7), against ≥ 6 of 7; fold ≥ 3 holds everywhere and the AUROC fails, for OPC on every sample, because the argmax OPC class holds the COP calls the v1 COP rule never emits as OPC. With the COP rule applied (b) only P1212 fails (2/7, 5/7); with its SEA-AD rescue (b′) or on WHB-only confident calls (c) only P1212_MERSCOPE fails (2/7 and 4/7). The user picks the definition in the M3 PR; a P1212_MERSCOPE failure under any option needs a fix or a written, user-approved exception at gate H.
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

**Rules.** Thresholds are fixed **now** and copied into `docs/acceptance/annotation-v1-preregistration.md` at M3 (the MO10 rule variant at M6b). Baselines may only **tighten** a threshold; loosening needs your written approval in the gate PR (R-sci). **Held out:** P7113 and P5011 (donors), reseg (segmentation, P7513 / P1212), the MERFISH AP series (MO10; region shares recomputed leaving each section and its neighbours out) and, for gate P, the simulation replicates on which frozen thresholds are tested (other held-out donors or test draws and seeds, NP4); none is used for tuning beyond the pre-registered selection rule, and their results are reported separately. Metrics come from `acceptance_metrics.json` and the acceptance scripts, on proseg_hybrid unless stated; [L] = first-run target.

### Human (gate H, at M8)

| ID | Criterion | Threshold | Basis |
|---|---|---|---|
| H1 | Soft broad JSD (7 classes, all table cells, whole section; 95% block-bootstrap CI and shared-mask value reported) | P7513 ≤ 0.17; P1212 ≤ 0.17; **P7113 ≤ 0.17; P5011 ≤ 0.31** (held out) | Soft .139 / .129 / .136 / .279 + 0.03 (`review_scientific/confident_jsd_bias_extra.txt`, E2 `cells_native`); E1 `real_jsd.csv` (iii) .132 / .143 under its scheme, so P1212 gets the larger margin [M]; legacy .51 / 1.00 / .66 / .44 |
| H2 | `flag_implausible` share of table cells | ≤ 1% per dataset | E1 (iii) 0.3–0.6% |
| H3 | WHB–SEA 7-class agreement, table cells ≥ 20 counts | P7513 M / X ≥ 0.90; P1212 M / X ≥ 0.80; P7113, P5011 ≥ 0.80 | E1 .932 / .917 / .826 / .904 |
| H4 | Held-out-gene enrichment (§5.8), per platform | Fold ≥ 3 and AUROC ≥ 0.70 in ≥ 6 of 7 broad classes on P7513, P1212, P7113 [L] | Non-circular (R-sci) |
| H5 | COP control | Confident COP supercluster ≤ 2%; confident broad OPC ≤ 10%; COP-derived OPC share reported | E1; `review_scientific/cop_demotion_rule.csv` |
| H6 | Simulation precision of the v1 raw thresholds (archived E2 HO contaminated simulation, reweighted to each dataset's composition) | Broad ≥ 0.90 at D ≥ 15; supercluster ≥ 0.85 at D ≥ 30, classes with n ≥ 50 [M] | Replaces the draft's circular curve test; H18 recomputes it in production |
| H7 | Confident broad coverage of table cells | P7513 M ≥ .62, X ≥ .44; P7113 M ≥ .67, X ≥ .39; P1212 M ≥ .34, X ≥ .29; P5011 M ≥ .30, X ≥ .40 | E2 LL-rule coverage − 0.10 (SEA-rule cost unmeasured until M3) |
| H8 | Gate levels | P1212_M and P5011_M `broad_only` (P5011_M with the warning flag); none `failed`; the rest `full` (mechanical, given A). Real-data QC (§8.8) only warns on these seeded datasets before this gate; its expected warnings are listed in the gate PR (e.g. P5011's paired broad JSD, soft .279 > 0.20; the marker-referee thresholds were measured only on P7513 and P1212, E1) | §5.4, §8.8 |
| H9 | Canonical-marker plausibility of confident `ct_broad` (marker pseudo-labels, `data/P7513` §C, `data/P1212` §4) | P7513 M ≥ 0.90, X ≥ 0.83; P1212 M / X ≥ 0.75 [M] | Legacy 69.6% / 44.2%; MMC at bp ≥ 0.9 93% / 85% |
| H10 | Marker referee on new-vs-legacy broad disputes | Markers side with the new label in ≥ 70% | E1 referee method |
| H12 | Cortical depth from manual pia/WM boundaries (label-independent) | P7513, P7113: Upper-layer IT < Deep-layer IT < Deep-layer NP/CT/6b medians with non-overlapping 95% CIs on both platforms. **All 4 pairs incl. P1212, P5011, at broad level:** oligodendrocytes WM > GM on both platforms [L] | R-sci; outputs exist for all 4 pairs × 2 platforms [H] |
| H13 | Contracts | MENDER completes on all 4 pairs under both unassigned policies; cortical-depth violins; H5AD and zarr id sets identical; legacy tables untouched before the flip (sha256); viewer on reseg (manual) | Wiring map §c |
| H14 | Resources | Annotation wall ≤ 2.5 h per pair (all segmentations); MAP peak RSS ≤ 16 GB; PREP ≤ 2.5 h per validated panel incl. resolvability; no GPU | §10 |
| H15 | Reproducibility | Identical re-run → identical DataFrame content; seed 0 vs 1 changes ≤ 1% of confident broad labels | E5: 99.85% stable at bp ≥ 0.8 |
| H16 | Flag transparency | Realised rates for every flag × class × platform; strata > 15% marked uninformative (mechanical) | §5.6 |
| H17 | Segmentation hold-out (reseg, P7513 / P1212) | H1 thresholds + 0.02; H2; H9 | R-sci |
| H18 | (rev2) Resolvability regression on the validated panels, from production PREP | Human set a (post-M0e): broad and supercluster emitted for every class with ≥ 50 test cells at depths from its v1 floor plus one grid step up to D_max; WHB cluster not emitted; simulated precision at 0.73 / 0.69 ≥ 0.90 / 0.85 at D ≥ 15 / 30 (local rule). ag7, VZG2: class and subclass emitted at min(dataset median depth, D_max(c)) for every class with ≥ 50 test cells. All three `validated`. No validated threshold would be raised, or each exception is listed in the gate PR | §8.3; E2; mouse report §4.2 |

**Human flip rule:** H1–H3, H5, H7–H10, H12–H16 on P7513 and P1212; H1–H3, H7, H8, H12 on P7113 and P5011; H4 on P7513, P1212, P7113; H6 and H18 once; H17 on reseg. Any failure needs a fix or a written, user-approved exception in the gate PR. The draft's split-rule criterion (H11) belongs to M11.

### Mouse (gate M, at M9)

| ID | Criterion | Threshold | Basis |
|---|---|---|---|
| MO1 | Class marker consistency over **all marker-pseudo-confident cells** (fixed denominator; confident-cell value reported) | ag7 proseg_hybrid ≥ 0.89; VZG2 original_seg ≥ 0.84 | .911 (`review_feasibility/mo1_ag7_marker_consistency.csv`), .856 (`review_scientific/mouse_vzg2_marker_consistency_panelLK.csv`) |
| MO2 | Pruning mechanics (not validation) | T2 after pruning ≤ 0.1%; class changes outside dropped nodes ≤ 0.25%; subclass ≤ 1.5% | E7 |
| MO3 | Inferred regions | = manual sets for ag7 and VZG2 | E7 §4 |
| MO4 | Soft class composition vs AP-matched MERFISH | VZG2 original_seg Immune ±0.5 and Astro-Epen ±3 points (MERFISH .31–.33: 1.50–1.56 / 17.2–17.5); ag7 ±1 / ±5 [L; age caveat] | Replaces the draft's circular MO4 / MO6 (R-sci) |
| MO5 | Spill-over flag | VZG2 held-out microglial-gene enrichment ≥ 10×; astrocyte FPR ≤ 0.5%; never shown as a count | E3 (35–40×) |
| MO6 | F1 rate after pruning | 0.2–1.0% | E7 |
| MO7 | Mouse gate | Invalid VZG2 proseg_hybrid `failed`; VZG2 original_seg and ag7 not `failed` | §7.6 |
| MO8 | Resources | ≤ 1.25 h per 4-segmentation section; PREP ≤ 2 h per panel incl. resolvability; MAP peak ≤ 12 GB; PREP peak ≤ 40 GB at 8 processes (query markers measured 20–27 GB at 16) | §10, §8.7 |
| MO9 | VZG2 after the M0a re-run | Registration passes; reseg / proseg_hybrid soft class JSD vs original_seg ≤ 0.05 [L] | Gates the *use* of those outputs, not the flip |
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
- **Dry run.** Before any new family is scored, the programme runs on the seeded families (set a, ag7, VZG2). They must pass at broad (human) / class (mouse) for every C_P class, and at supercluster / subclass for the classes H18 expects. If they do not, the criterion (statistic, n_min or margin) is unattainable rather than the panel bad; it is revised in a PR you approve (loosening rule, §14 Rules), and the revision is recorded before any new family is scored.

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
| `runs/<pair>/<seg>/` | `merxen annotate` outputs: `map_manifest.json`, `panel/`, `<plat>/<sid>_mmc_<run>.parquet`, `<plat>/<sid>_ct_provisional.parquet`, `logs/`; for P1212 / P5011 reseg also `panel_varderived/` (the refused var-derived panel) and `inputs_view/` |
| `logs/`, `PROGRESS.txt` | Run logs with `/usr/bin/time` wall and max RSS; `metrics.log` |

## 11. Shadow programme items 2–7 (M3 stage C2)

Measured 2026-09-27 with the same MAP configuration as §2 (no engine change); re-measured after the M3 review with matched-tile CIs, E2's SEA-AD definitions and the float32 threshold tolerance (header). Reports: `$A/shadow/<item>/REPORT.txt`; summary: `$A/shadow/SHADOW_SUMMARY.txt`. Code: `merxen.annotation.likelihood` (LL (vii)), `merxen.annotation.shadow` (additions below), `scripts/acceptance/{shadow_e8,shadow_x1,heldout_genes,shadow_flags,shadow_ll,shadow_glial_jsd}.py`. P7113, P5011 and reseg stay held out: they are reported, and nothing below was selected on them.

### 11.1 E8 segmentation comparison (OD-B6, OD-B7)

P1212 on all four segmentations (both platforms) and ag7 on all four (MERSCOPE, WMB ag7 bundle `d2e3f496`). Confident cells per mm² use the median tissue area of the section's segmentations (100 µm bins with ≥ 3 table cells). Foreign fraction = counts on the assigned class's negative genes / total counts. Mouse class confidence counts a class bp of exactly 0.90 (float32 tolerance, §2), which adds about 1% of cells to every segmentation.

| Sample | Metric | original_seg | proseg_mask | proseg_hybrid | reseg |
|---|---|---|---|---|---|
| P1212_MERSCOPE | Confident broad per mm² (ratio vs proseg_hybrid) | 654 (1.06) | 609 (0.99) | 616 (1.00) | 352 (0.57) |
| P1212_XENIUM | Confident broad per mm² (ratio) | 1,048 (0.95) | 1,091 (0.99) | 1,099 (1.00) | 908 (0.83) |
| P1212 | Soft broad JSD, MERSCOPE vs Xenium [95% CI, matched tiles] | 0.153 [0.146, 0.161] | 0.130 [0.123, 0.140] | 0.133 [0.126, 0.142] | 0.125 [0.110, 0.146] |
| P1212_MERSCOPE / _XENIUM | Foreign (negative-gene) fraction of confident cells | 0.022 / 0.022 | 0.019 / 0.022 | 0.019 / 0.021 | 0.005 / 0.012 |
| ag7 | Confident class / subclass per mm² (ratio) | 2,094 / 1,677 (1.11 / 1.05) | 1,877 / 1,584 (1.00 / 0.99) | 1,883 / 1,593 (1.00 / 1.00) | 2,086 / 1,786 (1.11 / 1.12) |
| ag7 | Foreign fraction; confident microglia | 0.0137; 136 | 0.0132; 255 | 0.0128; 294 | 0.0100; 1,114 |

- **OD-B6: the switch criterion is not met.** No segmentation has ≥ 25% more confident broad cells per mm² on P1212_MERSCOPE (best: original_seg, +6%), and all four leave it `broad_only` (A 0.21–0.26). The adopted policy (proseg_hybrid, broad level only) stands.
- **OD-B7, recommendation for the user:** per dataset. Human: keep proseg_hybrid (proseg_mask is equivalent; reseg has purer cells and a lower platform JSD but 17–43% fewer confident cells per mm²; original_seg has the highest platform JSD). Mouse ag7: reseg (most confident subclass cells, class cells equal to original_seg, lowest foreign fraction, almost four times the confident microglia). VZG2 is re-checked after the M0a re-run.

### 11.2 X1 platform rescaling (OD-B9)

Per-gene factors against the reference pseudobulk of each dataset's cells with WHB bp ≥ 0.8, median-centred and capped at ±2 log2, then a WHB set-a re-map. The C2 run (2026-09-27 01:06–01:12) scored X1 on a pass rule with a referee condition; that rule was written into `scripts/acceptance/shadow_x1.py` but committed after the run, and the logs recorded no script hash, so it cannot be shown to have been fixed before the runs. It is not used here. The decision below rests on effect size; the review rerun's launcher log records the script's sha256.

| Pair | Soft JSD set a / set c / X1 | X1 − set a [paired 95% CI] | Set c − set a [paired 95% CI] | Referee X1-vs-set-a disputes, X1 / set a wins (MERSCOPE; Xenium) | Referee set-c-vs-set-a disputes, set c / set a wins (MERSCOPE; Xenium) |
|---|---|---|---|---|---|
| P7513 | 0.129 / 0.121 / 0.128 | −0.001 [−0.002, −0.001] | −0.008 [−0.010, −0.006] | 0.37 / 0.40; 0.24 / 0.63 | 0.30 / 0.49; 0.23 / 0.65 |
| P1212 | 0.133 / 0.109 / 0.126 | −0.007 [−0.008, −0.006] | −0.024 [−0.035, −0.016] | 0.25 / 0.40; 0.21 / 0.58 | 0.20 / 0.53; 0.23 / 0.57 |
| P7113 (held out) | 0.126 / 0.110 / 0.126 | 0.000 [−0.000, 0.001] | −0.016 [−0.018, −0.014] | 0.33 / 0.38; 0.26 / 0.54 | 0.28 / 0.49; 0.26 / 0.58 |
| P5011 (held out) | 0.227 / 0.161 / 0.217 | −0.010 [−0.010, −0.009] | −0.066 [−0.069, −0.062] | 0.37 / 0.33; 0.28 / 0.53 | 0.28 / 0.44; 0.26 / 0.55 |

(Paired CIs over matched tiles, §2; the independent-resampling sensitivity is in `x1_jsd_differences.csv`.)

**Decision: X1 is not adopted** (OD-B9). On the development pairs its soft-JSD gain over set a is at most 0.007 (0.001 and 0.007), 18% and 29% of set c's gain (0.008 and 0.024), and H4 is unchanged (§11.3). The E1 referee is **not neutral** for rescaled labellings: it scores raw panel counts, so it favours labellings mapped from unrescaled counts. It sides with set a against set c as well (set c wins 0.20–0.30 of their disputes, set a 0.44–0.65), yet set c is the mandatory sensitivity, so the referee outcome is reported but not used as a criterion. `annotation_xplat_sensitivity` stays `geneset_c`; plan §17 keeps per-platform rescaling as a v2 candidate. The factors have a spread of 2.4–3.0 log2 and 36–50% of genes hit the cap, because they mostly measure reference (snRNA-seq) vs in-situ efficiency; their Xenium-minus-MERSCOPE difference correlates 0.80–0.92 with the measured cross-platform ratios (`research/xplat`).

### 11.3 H4 held-out-gene baseline

Metric as §2, with the "assigned class" left to the user (options (a)–(c) below). Markers on set a: Neurons SLC17A7, GAD1 and SLC17A6 (MERSCOPE) or GAD2 (Xenium); Astrocytes AQP4, GJA1; Oligodendrocytes MOBP, MOG, OPALIN; OPC PDGFRA, VCAN; Microglia CX3CR1, GPR34 (+ P2RY12 on Xenium); Vascular FLT1, PECAM1, ABCC9; Fibroblasts DCN, FBLN1, COL12A1 (`$A/shadow/heldout/heldout_markers.csv`). The asset's rank-2 alternates were added in M3 because set a lacks SATB2, PLP1, CSF1R, CLDN5 and LUM.

| Sample | (a) argmax, all table cells (M3 working metric) | (b) argmax with the COP rule | (b′) (b) with the SEA-AD OPC rescue | (c) WHB-only confident | Classes failing under (a) |
|---|---|---|---|---|---|
| P7513_MERSCOPE | 6 | 7 | 7 | 7 | OPC |
| P7513_XENIUM | 6 | 6 | 7 | 7 | OPC |
| P1212_MERSCOPE | 2 | 2 | 2 | 4 | Neurons, Oligodendrocytes, OPC, Microglia, Vascular |
| P1212_XENIUM | 5 | 5 | 6 | 7 | OPC, Microglia |
| P7113_MERSCOPE | 5 | 6 | 6 | 6 | OPC, Microglia |
| P7113_XENIUM | 6 | 6 | 7 | 7 | OPC |
| P5011_MERSCOPE (not scored by H4) | 0 | 2 | 2 | 4 | all |
| P5011_XENIUM (not scored by H4) | 5 | 5 | 6 | 7 | OPC, Microglia |

Classes passing of 7 (H4 needs ≥ 6 on P7513, P1212 and P7113, per platform; `$A/shadow/heldout/heldout_h4.csv`). Fold is ≥ 3 in every class and sample under every option; the AUROC decides.

- **Why OPC fails under (a).** The argmax OPC class holds every COP argmax call, which the v1 COP rule never emits as broad OPC (§5.2). OPC's AUROC is 0.54–0.66 under (a) on every sample; it is 0.60–0.89 under (b), 0.68–0.88 under (b′) and 0.70–0.95 under (c). Under (a) OPC fails by construction, so "≥ 6 of 7" becomes "all six other classes must pass".
- **"The assigned class" is an open choice for the user**, to be settled in the M3 PR before the definition is frozen; plan §5.8 does not define it, and M3 did not pre-register a reading. The options: (a) the argmax (strictest; P1212 both platforms and P7113_MERSCOPE fail); (b) the argmax with the §5.2 COP rule applied, COP calls that fail it staying at lineage and leaving the OPC class (P1212 both platforms fail); (b′) the same with the rule's SEA-AD confident-OPC rescue, whose SEA-AD calls saw the held-out genes (P1212_MERSCOPE fails); (c) WHB-only confident calls (P1212_MERSCOPE fails). Every option but (a) scores fewer cells than all table cells, so whichever the user picks is recorded as a definition in §2 before gate H, not as a loosening decided after the fact.
- Set c scores one class fewer (AQP4 is a set-c exclusion) and passes one class fewer on six of the eight samples under (a).

### 11.4 H16 prototype: realised flag rates

Confident cells, proseg_hybrid (`$A/shadow/flags/flag_rates.csv`). **Contamination:** 0.0–3.9% in every human class × platform stratum, 0.7–1.8% on ag7. Every stratum is informative except P5011_MERSCOPE fibroblasts, which have too few cells for a null. **Diffuse:** 0–44%. Ten of the 56 human strata exceed 15% and five exceed 30%: P7513_XENIUM oligodendrocytes 35% and microglia 44%; P7113_XENIUM oligodendrocytes 33%, microglia 35% and fibroblasts 30%. On ag7 the oligodendrocyte lineage is at 42%. These strata are marked uninformative. reseg is at ≤ 3.5% (contamination) and ≤ 4.4% (diffuse). Production flags are M4. H16's 15% and §4.3's 30% for the diffuse flag should be reconciled there.

### 11.5 LL value test (OD-B8) and the OD-B13 trigger

**The typer is the LL (vii) recipe on WHB-frontal bundle profiles**, not E1 (vii) itself: E1 `14_ll_contam.py` (`LLmix_wbctx+platform`) ported onto the WHB set-a bundle's 122 cluster profiles (17 superclusters, sinks included), where E1 used a 204-cluster whole-WHB cortex reference (plan §3.2 `ll_whb_ctx_profiles`). It typed every table cell of the 16 samples, once with factors capped at ±2 log2 (M10) and once uncapped (E1). Its agreement with E1 (vii) on E1's four 30k pilot subsets (`$A/shadow/ll/ll_vs_e1_vii.csv`):

| Pilot subset | Broad agreement, uncapped / capped | Supercluster agreement, uncapped / capped |
|---|---|---|
| P7513_MERSCOPE | 0.834 / 0.828 | 0.822 / 0.797 |
| P7513_XENIUM | 0.773 / 0.752 | 0.759 / 0.734 |
| P1212_MERSCOPE | 0.746 / 0.727 | 0.721 / 0.688 |
| P1212_XENIUM | 0.753 / 0.717 | 0.732 / 0.692 |

On the same cells the port's broad call agrees with WHB's argmax about as often as E1 (vii)'s does (0.71–0.82 uncapped, 0.65–0.82 capped, vs 0.68–0.81 for E1 (vii)), so the OD-B13 direction below does not depend on the port. Scoring OD-B8 on (vii) proper would need a rerun on `ll_whb_ctx_profiles`, which the store does not hold yet (§7, open question).

**OD-B8 has no pre-registered operationalisation.** The plan asks whether LL moves referee outcomes by > 2 points or coverage by > 5 points; how LL would be used is not stated. Two readings are reported; neither was fixed in advance (the first was chosen in `scripts/acceptance/shadow_ll.py` at M3 C2). OD-B8 goes to the user in the M3 PR with both.

| Sample (proseg_hybrid, capped) | (1) Coverage v1 → v1.1 (SEA-AD or LL below 60) | Bound: v1 without the below-60 rule − v1 | H10 v1 → v1.1 (ceiling gain) | Marker plausibility v1 → v1.1 | (2) LL as tie-breaker: referee gain | E2 LL rule − v1 | LL as the below-60 vote − v1 |
|---|---|---|---|---|---|---|---|
| P7513_MERSCOPE | 0.704 → 0.707 | 0.005 | 0.989 → 0.987 (+0.000) | 0.936 → 0.935 | +0.096 | −0.030 | −0.020 |
| P7513_XENIUM | 0.597 → 0.598 | 0.002 | 0.980 → 0.979 (+0.000) | 0.907 → 0.907 | +0.125 | −0.047 | −0.039 |
| P1212_MERSCOPE | 0.438 → 0.443 | 0.009 | 0.961 → 0.959 (+0.001) | 0.940 → 0.937 | +0.061 | −0.054 | −0.043 |
| P1212_XENIUM | 0.510 → 0.512 | 0.003 | 0.978 → 0.977 (+0.000) | 0.925 → 0.925 | +0.077 | −0.086 | −0.079 |
| P7113_MERSCOPE | 0.756 → 0.759 | 0.005 | 0.991 → 0.990 (+0.000) | 0.954 → 0.954 | +0.141 | −0.037 | −0.023 |
| P7113_XENIUM | 0.648 → 0.650 | 0.002 | 0.897 → 0.895 (+0.002) | 0.936 → 0.935 | +0.093 | −0.063 | −0.052 |
| P5011_MERSCOPE | 0.459 → 0.466 | 0.009 | 0.940 → 0.937 (+0.002) | 0.940 → 0.937 | +0.112 | −0.056 | −0.035 |
| P5011_XENIUM | 0.556 → 0.559 | 0.004 | 0.926 → 0.924 (+0.002) | 0.924 → 0.924 | +0.097 | −0.047 | −0.035 |

- **Reading (1), LL as the v1.1 below-60 vote: not met, and uninformative.** v1.1's confident cells (SEA-AD or LL agreeing below 60 counts) are a subset of "v1 without the below-60 rule" (the same ≥ 60 veto and rescues; checked on every sample, `v11_within_bound`). So the coverage gain is bounded by the below-60 SEA-AD cost, 0.2–0.9 points here and at most 1.0 on any dataset, far below 5; measured +0.1 to +0.6. H10 is at 0.90–0.99 with 10,000–51,000 disputes and v1.1 adds 204–561 cells, so even if every added cell were a won dispute H10 would rise by at most 0.2 points (`h10_ceiling_gain`), below 2. The test cannot pass by construction; its outcome says nothing about LL. The uncapped factors give the same.
- **Reading (2), LL as tie-breaker of WHB-vs-SEA-AD disputes: exceeds 2 points, but circular.** With LL deciding the disputes, markers side with the chosen label 6.1–12.5 points more often than with WHB alone on the development datasets (6.1–14.1 on all, capped; 7.7–14.5 uncapped). LL and the referee read the same marker counts, so this favours LL by construction, and it is not a v1.1 behaviour (WHB names every class, OD-B2).
- **For the user (M3 PR):** keep LL out of v1.1 (reading 1; `likelihood.py` stays for M10), or adopt it as a tie-breaker (reading 2) with a non-circular check first (held-out genes, §5.8), or score OD-B8 on (vii) proper once `ll_whb_ctx_profiles` is built.
- **OD-B13: not triggered.** The below-60 SEA-AD rule removes 0.2–0.9 points. The LL rules keep 0.6–8.6 points *less* coverage than the SEA-AD rule on every dataset (reseg included, both factor variants). The LL part of M10 stays after M8.
- For information: in WHB-vs-LL disputes (proseg_hybrid, capped) the referee is split, with WHB winning 0.36–0.60 of them and LL 0.22–0.45.

### 11.6 WHB vs SEA-AD glial JSD (information, OD-B2 decided)

Glial (astrocytes, oligodendrocytes, OPC, microglia) soft JSD, MERSCOPE vs Xenium, WHB − SEA-AD with a paired block-bootstrap CI (matched tiles, one draw for both sections and both references; §2), SEA-AD's soft composition on E2's class × subclass mass (§2): P7513 −0.025 [−0.027, −0.023], P1212 −0.038 [−0.047, −0.029], P7113 −0.021 [−0.023, −0.019], P5011 −0.061 [−0.064, −0.057]. WHB's glial composition agrees better across platforms on every pair. At 7 classes (soft) WHB is also better on every pair (−0.013, −0.032, −0.005, −0.016); before the review SEA-AD's soft mass lacked the class-level factor, which made its composition harder than WHB's root-level one and gave P5011 +0.011. The argmax 7-class difference on P5011 stays +0.031 [0.026, 0.037]. No decision is asked for.

### 11.7 Files

| Directory under `$A/shadow/` | Content |
|---|---|
| `e8/` | `runs/` (P1212 original_seg and proseg_mask, ag7 original_seg, proseg_mask and reseg MAP outputs), `metrics/e8_{human,human_jsd,mouse}.csv`, `REPORT.txt` |
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

**Re-measured** (resolvability version 3; `$A/m3b/followup/H18_FOLLOWUP.txt`): set a PREP H18 classes failing (broad / supercluster) 2 / 1 under version 2, 1 / 0 with decision 1 alone (broad OPC at 15), 2 / 1 with both decisions (broad Astro at 120 on power; broad and supercluster Oligo at 120, where other-region COP cells are called Oligodendrocyte); Vascular and Fibroblast are now emitted. VZG2 passes; ag7 keeps the Immune subclass at 100 counts and, reweighted, the classes its datasets hardly contain. The H7 proxy (confident broad share after reweighting, an upper bound of H7) passes on every dataset with decision 1 alone and misses on P7513 MERSCOPE with both (.576 against ≥ .62; .638 with decision 1 alone). These are the H18 / H7 exceptions for the gate PRs; no threshold is changed.

## 13. Decisions of 2026-09-28 (M3b final follow-up and review 3)

M3b review 2 (resolvability version 4, `$A/m3b/review2/REVIEW2_H18.txt`) left two open points: 694 of the 2,955 other-region test cells were of clusters the held-out training reference lacks, and the rebuilt bundles redrew the simulation of every test cell because one random stream ran over all of them (0–7 of 85 validated broad / supercluster bins per human dataset moved; P5011 MERSCOPE's H7 proxy went from .311 to .297, below its .30 target). Two changes follow, neither of which changes a threshold, a target or a decision rule:

1. **Other-region cells only of training clusters** (user decision 2026-09-27, implemented 2026-09-28; plan OD-E11). The human held-out test set's other-region non-neuronal top-up draws only from clusters the held-out training reference keeps (≥ 5 training cells). The candidates left out are counted in `bundle.json` (`test_set.other_region`: `cluster_rule`, `n_excluded_cluster_not_in_training` per supercluster, and a check that no drawn cell is of an absent cluster). `HO_OTHER_REGION_VERSION` 2 enters the held-out and self-map build hashes.
2. **Keyed simulation draws** (robustness fix, resolvability version 5, corrected in version 6; plan §8.3 step 3). Each simulated cell's thinning and spill thinning come from a generator seeded by a stable hash of the seed, the recipe name and version, the cell id and the depth, and the spill partner is chosen by rendezvous hashing (uniform over the eligible cells). Adding, removing or reordering test cells then leaves every other simulated cell unchanged, except that a host whose partner leaves (or is outranked by an added cell) gets a new spill. The per-gene efficiency stays the pre-registered draw (one generator seeded by `[0, 0x6566]` over the panel's gene order), which never depended on the test cells. *Correction (M3b review 3, 2026-09-28):* the version-5 build had also keyed each gene's efficiency by the gene id. Neither change needed that, and it replaced the pre-registered seed-0 realisation with an unrelated one (correlation of the log efficiencies −0.06 on set a); resolvability version 6 restores the pre-registered draw and keeps the per-cell keys. With it the recipe (LogNormal(0, 0.8) efficiency in its seed-0 realisation, 25% spill, seed 0) is unchanged.

**Re-measured** (resolvability version 6; `$A/m3b/final_followup/FINAL_FOLLOWUP.txt` §12, `$A/m3b/final_followup/review3/H18_V6.txt`, `EXCEPTIONS_V6.txt`). The set a test set holds 11,040 cells (8,777 donor + 2,263 other-region; Vascular 412, Fibroblast 144, COP 152). Set a PREP fails H18 at broad Oligo 120 counts (79 calls, precision .911, Wilson bound .828 < .88); version 4 failed broad Astro, broad and supercluster Oligo, and supercluster Astro at 120. Reweighted to the eight human datasets, 23 (level, class) pairs fail against 25 under version 4: broad OPC on all eight (at 15 counts, and at 120 on five), broad Oligo on seven and supercluster Oligo on six (other-region COP cells of training clusters called Oligodendrocyte, 9–26% of the called set's confident weight), and broad and supercluster Immune at 60 on P5011 MERSCOPE. VZG2 PREP passes; ag7 PREP fails the Immune subclass at 100 in this draw (as in version 4; it passes with seeds 1 and 2); reweighted, ag7 fails 0–2 classes or subclasses at 250 counts per segmentation (05 OB-IMN GABA, 09 CNU-LGE GABA; few positive-weight calls or small Kish n). The H7 proxy (confident broad share after reweighting, an upper bound of H7) is P7513 M .683, X .605; P7113 M .739, X .656; P1212 M .354, X .523; P5011 M .298, X .557: it misses on P5011 MERSCOPE (≥ .30 asked), as under version 4 (.297). These are the H18 / H7 exceptions for the gate PRs; no threshold is changed.

**Version 5's numbers are not the outcome of these changes.** With its redrawn efficiency, 15 reweighted pairs failed, set a PREP passed and the H7 proxy passed on all eight datasets (P5011 MERSCOPE .380). Those results come from the unrequested redraw; they are one more draw of the recipe, not the result of the two changes, and the gate PRs should not cite them as passes.

**The verdicts depend on the simulation draw** (`$A/m3b/final_followup/review3/FACTORIAL_ANALYSIS.txt`). A 3 × 3 grid on set a crossed the per-cell seed (thinning, spill, partner) with the efficiency seed, one realisation per grid cell, mapped at seed 0; the additive model's residual holds the interaction and the mapping noise. Across the grid, P5011 MERSCOPE's H7 proxy spans .297–.431 and is below .30 in 2 of 9 cells, both with the seed-0 efficiency; P1212 MERSCOPE spans .320–.444 (below .34 in 1); the other datasets stay at least .028 above target. The reweighted H18 failures number 12–26 and set a PREP's 2–8. Changing only the efficiency seed moves on average 6.3 of the 83 validated broad / supercluster bins per dataset (1–14); changing only the per-cell seed moves 4.4 (0–9), both 6.5. Mapping the same simulated cells in another order (MapMyCells draws its bootstrap per chunk of query cells) moves 1.4 (0–2), yet that alone lifts P1212 MERSCOPE's proxy from .354 to .433 and adds 4 set a PREP failures. For the H7 proxy and the H18 counts neither factor is significant in this grid (F(2, 4) below the .05 critical value 6.94, except P1212 Xenium's H7 proxy, whose range is .007), and the residual is the largest component for most metrics. Only broad OPC at 15 clearly follows the efficiency draw: it fails on all eight datasets with the seed-0 efficiency under every per-cell seed, and on 0–5 with seeds 1 and 2 (F = 18.1, p = .01). The version-5 statement that the single efficiency draw moves these verdicts more than the per-cell draws do therefore holds for broad OPC at 15 and, on average, for the number of bins that move; it does not hold for the H7 proxy or the H18 counts. How the gate treats this spread (the pre-registered seed-0 draw, several draws, or the worst of them) is a pre-registration choice left to the user; until then no H18 / H7 verdict is recorded as a pass.

## 14. Decisions of 2026-09-27 (after the M3 and M3b PRs; recorded at M4)

These decisions change no threshold, target or decision rule in §9. Plan: OD-B7, OD-B8 and OD-B14 in "Open decisions for the user", D-A5, §7.4 and the §14 human flip rule.

1. **Default segmentation (OD-B7): proseg_hybrid for human, reseg for mouse.** This is the M3 E8 recommendation (§11.1): human keeps proseg_hybrid (reseg has purer cells but 17–43% fewer confident cells per mm²); mouse ag7 uses reseg (most confident subclass cells, lowest foreign fraction). The registered criteria keep the segmentations they name (§9: human on proseg_hybrid with reseg held out for H17; MO1 on ag7 proseg_hybrid and VZG2 original_seg); the decision sets the pipeline default, not a criterion. VZG2 is re-checked after the M0a re-run.
2. **No likelihood-typer vote in v1.1 (OD-B8).** LL does not join the v1.1 consensus in either reading of §11.5. `merxen.annotation.likelihood` and the LL (vii) recipe stay for diagnostics. The §5.3 degraded-mode truth table keeps its LL rows (`consensus.DEGRADED_MODES`), but no production configuration enables them. OD-B13 stays not triggered (§7 item 2).
3. **H2, H4 and the H7 / H18 exceptions are decided at M8 (OD-B14).** The baseline H2 failures (P5011_MERSCOPE proseg_hybrid 1.02%, P1212_MERSCOPE reseg 1.04%; §7 item 1), H4's open "assigned class" and its P1212 failures (§7 item 7, §11.3), and the H7 / H18 exceptions of §12–§13 (P5011 MERSCOPE's H7 proxy .298; the reweighted H18 failures) are re-measured at M8 with M4's final RESOLVE rules: statuses, resolvability-gated emission reweighted to each dataset, and the COP rule. The M8 gate PR brings, for each, a fix or a written exception proposal for the user's approval. Until then no H2, H4, H7 or H18 verdict is recorded as a pass, and no threshold is loosened (§1).
