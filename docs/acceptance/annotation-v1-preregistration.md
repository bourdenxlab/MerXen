# Annotation v1 pre-registration: acceptance thresholds and M3 baselines

- **Effort:** robust cell-type annotation (`robust-celltype-annotation`), milestone M3 (plan §12 "M3", item 1 of the shadow programme).
- **Plan:** `docs/plans/robust-celltype-annotation-plan.md` rev3; the thresholds in §9 below are its §14, copied verbatim (plan file sha256 `14804ff24dcd46ab7d649775bb25aba1b3bac08d533ddd93d457a87f68895024`, last changed in `06a04a1`).
- **Measured:** 2026-09-27, branch `feature/rca-m3-mmc-engine` (MAP engine `2f346b9`, `96f2834`; shadow metrics `b4a71fe`; baseline script `ed373e5`).
- **Evidence root** (`$A`): `/srv/storage/MerXen/annotation_dev/evidence_20260926`. Every baseline below is in `$A/shadow/baselines/metrics/*.csv`; §10 lists the files.

## 1. Rules

1. **The thresholds are fixed by this document.** They are the plan's §14 values, copied verbatim in §9. No threshold is changed here.
2. **Baselines may only tighten a threshold.** A measured baseline can justify a stricter threshold, decided in the M3 PR. Loosening a threshold needs the user's written approval in the gate PR. That includes redefining a metric so that it becomes easier to pass, or dropping a dataset, a class or a criterion. This applies to gate H (M8), gate M (M9) and gate P.
3. **Held-out validation sets are reported separately and are not used for tuning:** the donors P7113 and P5011 (§4) and the second segmentation, reseg (§5; H17 is scored on P7513 and P1212 reseg; P7113 and P5011 reseg are reported for information). The development datasets are P7513 and P1212 on proseg_hybrid (§3). Held-out results never select a rule variant, threshold or floor. The pre-registered selection rules (MO10's variant at M6b, gate P's evaluation rules) are the only exceptions.
4. **Metric definitions are the ones in §2.** RESOLVE (M4) and the acceptance scripts (M8) must compute the gate metrics the same way or document the difference in the gate PR. A difference that makes a criterion easier to pass counts as a loosening (rule 2).

## 2. What was measured, and how

**Inputs (read-only).** These are the published `<pair>_<PLAT>_clustered.h5ad` of P7513, P1212, P7113 and P5011 × MERSCOPE / Xenium, on proseg_hybrid and reseg: `/srv/storage/MerXen/results/<pair>/<seg>/clustering_squidpy/clustering_squidpy_out/<plat>/`. The runs use `layers["counts"]` on the table cells (every object in the clustered file; `total_counts >= 10` after control removal). The shared tissue mask comes from `results/<pair>/alignment/align_out/shared_tissue_mask.npy`, in the Xenium frame of `registration_summary.json`. The MERSCOPE coordinates are already in that frame (shape key `*_aligned_nonrigid`). The segmented-object counts come from `$A/research/lowcount/qc_summary.csv`.

**MAP runs (M3 stage A code).** Each pair × segmentation ran `merxen annotate --species human --from-clustered-h5ad <M> --from-clustered-h5ad <X> --store /media/mathieubo/SSD1/MerXen/annotation_references --gene-id-fallback-csv <WHB-10Xv3-Nonneurons-raw.h5ad> --n-processors 6 --out $A/shadow/baselines/runs/<pair>/<seg>`, with at most three jobs at once. The engine settings were bootstrap factor 0.5, 100 iterations, seed 0, 6 processes, OMP / OpenBLAS / MKL / numba threads 1, raw normalisation, and cell_type_mapper 1.7.2 (`824caef`).
- The bundles were builder v2: WHB frontal set a `b70dc181` (panel `6e5fd5fb`, 297 genes, post-M0e), WHB frontal set c `f20d11b0` (panel `77bd80ed`, 265 genes, curated E5 family; proseg_hybrid only) and SEA-AD Multiregion `3973770c` (set a).
- **Declared panel for reseg.** The published reseg MERSCOPE `var` of P1212 (299 features) and P5011 (268) is `min_cells`-filtered. A panel derived from it would get a new hash that no bundle has. Plan §3.2 requires the declared panel instead. Those two pairs were therefore re-run through `merxen annotation-panel --panel-file <PLAT>=<same section's proseg_hybrid clustered H5AD>`: that file's unfiltered 300-feature `var` is the declared panel. The re-run gave set a `6e5fd5fb`, and `merxen annotate --panel-dir` then mapped the genes present. P1212 reseg MERSCOPE lacks 1 of the 297 genes. P5011 reseg MERSCOPE lacks 32 of 297 (10.8%). Both were mapped with a restricted lookup (M3 behaviour; §7 item 4).

**Shadow v1 rules** (`merxen.annotation.shadow.evaluate_human_rules`). RESOLVE is M4, so the confident statuses come from a shadow evaluation of plan §5.2 rules 1, 2 and 4 and the §5.4 gate. It uses the raw v1 thresholds (WHB broad / lineage 0.73, supercluster 0.69, SEA-AD broad 0.68) and the packaged set-a floors (`floors_human.csv`). There are three rules:
- **Lineage** needs aggregated bp >= 0.73, a plausible node (an implausible node keeps its lineage when SEA-AD agrees at lineage) and the second vote.
- **Broad** needs a confident lineage, aggregated bp >= 0.73, counts >= the class × platform broad floor, and the SEA-AD rule. Below 60 counts SEA-AD must agree at the 7-class level. From 60 counts SEA-AD must not confidently disagree. The COP rule applies: a COP call is broad OPC only with >= 120 counts and supercluster bp >= 0.69, or with a confident SEA-AD OPC call.
- **Supercluster** needs a confident broad, gate level `full`, bp >= 0.69 and counts >= the supercluster floor.

The rules have no resolvability (M3b) and no flags. M4's exit requires RESOLVE's coverage to be within ±0.05 of these baselines.

**Metric definitions.**

| Criterion | Operational definition (code) |
|---|---|
| H1 | Soft broad composition: per table cell, the bootstrap probability of the assigned WHB supercluster and of its 5 runner-ups, aggregated to the 7 broad classes through `whb_supercluster_vocab.csv`. Sinks and nodes outside the 7 classes, plus the residual 1 − Σ, are "unallocated" (`soft_matrix_from_provisional`). The composition is the sum over cells. The JSD is the base-2 Jensen-Shannon **distance** (as `scipy.spatial.distance.jensenshannon` and E1 `real_jsd.csv`) of the renormalised 7-class vectors, MERSCOPE vs Xenium. The 95% CI is a spatial block bootstrap: 500 µm square tiles, 200 replicates, each section's tiles resampled independently with replacement, seed 0, percentile interval (`tile_codes`, `tile_sums`, `block_bootstrap_jsd`). It is reported on the whole section and inside the shared tissue mask. Also reported: argmax (broad class of the assigned supercluster), confident-only (the v1 shadow confident broad label), the soft composition over cells with >= 30 counts, and set c. |
| H2 | Share of table cells whose assigned WHB supercluster is a sink (Miscellaneous, Splatter) or not region-plausible for frontal cortex (vocab), whatever its bp. This is `flag_implausible` with `implausible` status taking precedence over `low_confidence`. |
| H3 | Share of table cells with >= 20 counts where WHB's 7-class label (broad class of the assigned supercluster) equals SEA-AD's (subclass → broad class through `seaad_mr_subclass_vocab.csv`, "VLMC & Perivascular" split by supertype). As in E1, two labels outside the 7 classes agree; the strict variant, where they disagree, is in `sample_metrics.csv` and differs by < 0.001 on every dataset. |
| H5 | Confident COP supercluster share and confident broad OPC share of table cells (v1 shadow rules). The COP-derived OPC share is the share of confident broad OPC cells whose WHB call is COP. |
| H7 | Confident broad coverage of table cells under the v1 shadow rules. Coverage of segmented objects is also reported. |
| H8 | Gate on table cells: A = share with >= 30 counts; `broad_only` if A < 0.30; `failed` if confident broad coverage of table cells < 0.25; warning if coverage of segmented objects < 0.15 (`dataset_gate`). |
| H10 | E1 marker referee (`exp/E1/09_marker_referee.py`; `marker_class_scores`, `marker_referee`). For each disputed cell, where both labels are among the 7 classes and they differ, the label whose canonical panel markers (E1's list) hold the larger fraction of the cell's counts wins. "New" is the v1 shadow confident broad label, so cells that are not confident have no new label. "Legacy" is the published `obs["broad_class"]`. |
| OD-B13 | On E2's 30k native cells per dataset (`$A/exp/E2/out/cells_native.csv.gz`, the only cells with likelihood-typer calls; all 30k shared with the table), the confident broad coverage is computed with the same production WHB and SEA-AD calls under five variants: (a) v1 without the below-60 rule; (b) v1; (c) v1 with the likelihood typer instead of SEA-AD as the below-60 vote; (d) E2's LL rule alone (`exp/E2/final_coverage.py`: LL must agree below 60 counts, neurons merged, fibroblasts pooled with vascular cells; no SEA-AD role); (e) no second method. The below-60 SEA-AD cost is (a) − (b). The trigger (> 0.08 on any dataset moves LL before M8) is scored on (c) − (b) and on (d) − (b). |
| M3 exit | Production vs the E1 (iii) pilot (`$A/research/pilot/{whb_region,seaad_mr}`, P7513 and P1212 30k subsets). The comparison covers the WHB supercluster assignment on cells with production bp >= 0.8 (target >= 0.98) and the SEA-AD subclass assignment on all shared cells (target >= 0.945 − 0.02 = 0.925). P7113 and P5011 are compared with E2's runs of the pilot configuration (`$A/exp/E2/mmc/{WHBF,SEAAD}__<sid>_sub30k.json`), for information. |

**Code.** Metrics: `src/merxen/annotation/shadow.py` (tests: `tests/test_annotation/test_shadow.py`). Driver: `scripts/acceptance/shadow_baselines.py --runs-root $A/shadow/baselines/runs --results-root /srv/storage/MerXen/results --evidence-root $A --out $A/shadow/baselines/metrics` (75 s, 4 GB).

## 3. Development baselines: P7513 and P1212, proseg_hybrid

These are the rows that the §9 human flip rule scores for H1–H3, H5, H7–H10 and H12–H16.

| Pair | H1 threshold | Soft, whole section [95% CI] | Soft, shared mask [95% CI] | Argmax, whole | Confident, whole | Soft, cells >= 30 counts | Set c soft / argmax | Unallocated soft mass M / X |
|---|---|---|---|---|---|---|---|---|
| P7513 | <= 0.17 | 0.129 [0.095, 0.166] | 0.128 [0.093, 0.174] | 0.130 [0.097, 0.166] | 0.145 [0.100, 0.189] | 0.211 | 0.121 / 0.121 | 0.017 / 0.017 |
| P1212 | <= 0.17 | 0.133 [0.122, 0.154] | 0.154 [0.136, 0.183] | 0.149 [0.136, 0.174] | 0.206 [0.187, 0.232] | 0.183 | 0.109 / 0.117 | 0.019 / 0.020 |

| Sample | Table cells | Median counts | A (share >= 30 counts) | Gate level | Warning | Confident broad, table cells [H7 threshold] | Confident broad, segmented objects | Lineage | Supercluster | Implausible calls (H2 <= 1%) | WHB-SEA 7-class, >= 20 counts [H3 threshold] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P7513_MERSCOPE | 164,370 | 71 | 0.759 | full | no | 0.704 [>= 0.62] | 0.546 | 0.744 | 0.617 | 0.60% | 0.926 [>= 0.90] |
| P7513_XENIUM | 132,489 | 43 | 0.641 | full | no | 0.597 [>= 0.44] | 0.471 | 0.639 | 0.517 | 0.35% | 0.917 [>= 0.90] |
| P1212_MERSCOPE | 102,886 | 17 | 0.234 | broad_only | no | 0.438 [>= 0.34] | 0.152 | 0.597 | 0.000 | 0.30% | 0.825 [>= 0.80] |
| P1212_XENIUM | 157,676 | 25 | 0.426 | full | no | 0.511 [>= 0.29] | 0.332 | 0.587 | 0.446 | 0.36% | 0.908 [>= 0.80] |

| Sample | COP argmax calls | Confident COP supercluster (H5 <= 2%) | Confident broad OPC (H5 <= 10%) | COP-derived share of confident OPC | COP suppressed (`flag_cop_suppressed`) | New vs legacy referee: disputes (share); new / legacy / tie (H10 >= 0.70) | WHB vs SEA-AD referee: disputes (share); WHB / SEA-AD / tie |
|---|---|---|---|---|---|---|---|
| P7513_MERSCOPE | 4.14% | 0.27% | 2.36% | 0.628 | 0.80% | 51,022 (0.310); 0.988 / 0.006 / 0.007 | 17,683 (0.108); 0.395 / 0.339 / 0.267 |
| P7513_XENIUM | 2.80% | 0.11% | 1.56% | 0.635 | 0.47% | 20,680 (0.156); 0.980 / 0.007 / 0.013 | 13,667 (0.103); 0.466 / 0.370 / 0.164 |
| P1212_MERSCOPE | 8.90% | 0.00% | 1.48% | 0.829 | 3.39% | 43,545 (0.423); 0.961 / 0.006 / 0.033 | 22,384 (0.218); 0.298 / 0.314 / 0.388 |
| P1212_XENIUM | 4.18% | 0.02% | 1.21% | 0.710 | 0.98% | 24,674 (0.156); 0.977 / 0.008 / 0.015 | 19,822 (0.126); 0.369 / 0.365 / 0.266 |

- **H1.** The soft JSD is 0.129 (P7513) and 0.133 (P1212), against a threshold of 0.17. The plan's basis (E2 `cells_native`) was 0.139 / 0.129. The argmax JSD is 0.130 / 0.149, against E1 `real_jsd.csv` (iii) 0.132 / 0.143. The P7513 interval is wide (0.095–0.166) because the broad composition varies between 500 µm tiles, but its upper bound is below the threshold. Confident-only compositions are further apart (0.145 / 0.206; the draft floors gave 0.236 / 0.285), which is why H1 is soft (D-C5).
- **H2.** The implausible share is 0.30–0.60%, within the 1% threshold (E1 (iii): 0.3–0.6%). The calls go to Amygdala excitatory (region-implausible), Miscellaneous and Splatter (`implausible_top_nodes`).
- **H3.** Agreement is 0.926 / 0.917 / 0.825 / 0.908 (E1: 0.932 / 0.917 / 0.826 / 0.904).
- **H5.** Confident COP supercluster calls are at most 0.27% and confident broad OPC calls at most 2.4%. COP-derived calls make up 63–83% of confident OPC (plan: 62–86%).
- **H7.** Coverage clears its threshold by 0.08 to 0.22.
- **H8.** P1212_MERSCOPE is `broad_only` (A = 0.234). Its coverage of segmented objects, 0.152, is only 0.002 above the warning threshold (§7 item 3). Every other development sample is `full`.
- **H10.** Markers side with the new label in 96–99% of new-vs-legacy disputes. In WHB-vs-SEA-AD disputes the referee is close to a tie (E1: "roughly a tie").
- **OD-B13.** The below-60 SEA-AD rule costs 0.3–0.9 points of coverage. Replacing it with the likelihood typer would cost 2.0–5.8 points, so the LL variant keeps less coverage than the SEA-AD rule, not more. The trigger is not met.

Second vote on E2's 30k native cells (coverage of those cells; `second_vote_cost.csv`):

| Sample | Shared cells (share < 60 counts) | No second method | v1 without the below-60 rule | v1 (SEA-AD below 60) | v1 with LL below 60 | E2 LL rule alone | Below-60 SEA-AD cost | Below-60 LL cost | LL minus SEA-AD (trigger if > +0.08) |
|---|---|---|---|---|---|---|---|---|---|
| P7513_MERSCOPE | 30,000 (0.44) | 0.698 | 0.710 | 0.704 | 0.690 | 0.678 | 0.006 | 0.020 | -0.014 |
| P7513_XENIUM | 30,000 (0.61) | 0.597 | 0.606 | 0.603 | 0.551 | 0.543 | 0.003 | 0.054 | -0.052 |
| P1212_MERSCOPE | 30,000 (0.92) | 0.436 | 0.447 | 0.438 | 0.394 | 0.383 | 0.009 | 0.053 | -0.044 |
| P1212_XENIUM | 30,000 (0.83) | 0.500 | 0.509 | 0.506 | 0.451 | 0.443 | 0.003 | 0.058 | -0.055 |

M3 exit check against the E1 (iii) pilot (`exit_checks.csv`):

| Sample | Comparator | WHB supercluster, production bp >= 0.8 (n) [>= 0.98] | WHB supercluster, all shared cells | SEA-AD subclass, all shared cells [>= 0.925] | SEA-AD subclass, production bp >= 0.8 |
|---|---|---|---|---|---|
| P7513_MERSCOPE | E1 (iii) pilot | 1.0000 (16,644 of 30,000) | 0.9587 | 0.9573 | 0.9988 |
| P7513_XENIUM | E1 (iii) pilot | 1.0000 (13,625 of 30,000) | 0.9519 | 0.9453 | 0.9973 |
| P1212_MERSCOPE | E1 (iii) pilot | 1.0000 (9,722 of 30,000) | 0.9272 | 0.9483 | 0.9991 |
| P1212_XENIUM | E1 (iii) pilot | 1.0000 (11,275 of 30,000) | 0.9428 | 0.9413 | 0.9983 |

Both exit criteria hold on all four samples: the WHB supercluster agreement is 1.0000 at production bp >= 0.8, and the SEA-AD subclass agreement is 0.941–0.957 against 0.925. The third exit criterion, wall time within §10, is in §6.

## 4. Held-out donors: P7113 and P5011, proseg_hybrid (reported separately)

The flip rule scores H1–H3, H7, H8 and H12 on these datasets. E2 used thinned high-count cells of P7113 (both platforms) and P5011 Xenium in its calibration experiments (`exp/E2/common.py` `CALIB`), which is the evidence behind the v1 raw thresholds. Nothing in M3 was tuned on them.

| Pair | H1 threshold | Soft, whole section [95% CI] | Soft, shared mask [95% CI] | Argmax, whole | Confident, whole | Soft, cells >= 30 counts | Set c soft / argmax | Unallocated soft mass M / X |
|---|---|---|---|---|---|---|---|---|
| P7113 | <= 0.17 | 0.126 [0.092, 0.171] | 0.107 [0.075, 0.148] | 0.110 [0.074, 0.158] | 0.156 [0.111, 0.200] | 0.326 | 0.110 / 0.096 | 0.016 / 0.017 |
| P5011 | <= 0.31 | 0.227 [0.210, 0.243] | 0.235 [0.216, 0.254] | 0.253 [0.234, 0.272] | 0.324 [0.301, 0.343] | 0.243 | 0.161 / 0.183 | 0.038 / 0.030 |

| Sample | Table cells | Median counts | A (share >= 30 counts) | Gate level | Warning | Confident broad, table cells [H7 threshold] | Confident broad, segmented objects | Lineage | Supercluster | Implausible calls (H2 <= 1%) | WHB-SEA 7-class, >= 20 counts [H3 threshold] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P7113_MERSCOPE | 96,168 | 46 | 0.667 | full | no | 0.756 [>= 0.67] | 0.371 | 0.794 | 0.677 | 0.77% | 0.948 [>= 0.80] |
| P7113_XENIUM | 163,381 | 32 | 0.541 | full | no | 0.648 [>= 0.39] | 0.496 | 0.707 | 0.563 | 0.59% | 0.929 [>= 0.80] |
| P5011_MERSCOPE | 58,565 | 16 | 0.129 | broad_only | yes | 0.460 [>= 0.30] | 0.106 | 0.556 | 0.000 | 1.02% | 0.828 [>= 0.80] |
| P5011_XENIUM | 114,294 | 28 | 0.472 | full | no | 0.557 [>= 0.40] | 0.318 | 0.632 | 0.369 | 0.91% | 0.921 [>= 0.80] |

| Sample | COP argmax calls | Confident COP supercluster (H5 <= 2%) | Confident broad OPC (H5 <= 10%) | COP-derived share of confident OPC | COP suppressed (`flag_cop_suppressed`) | New vs legacy referee: disputes (share); new / legacy / tie (H10 >= 0.70) | WHB vs SEA-AD referee: disputes (share); WHB / SEA-AD / tie |
|---|---|---|---|---|---|---|---|
| P7113_MERSCOPE | 3.28% | 0.12% | 2.25% | 0.681 | 0.44% | 42,113 (0.438); 0.991 / 0.004 / 0.005 | 8,085 (0.084); 0.327 / 0.345 / 0.328 |
| P7113_XENIUM | 3.51% | 0.10% | 1.97% | 0.648 | 0.64% | 13,522 (0.083); 0.896 / 0.044 / 0.059 | 14,908 (0.091); 0.410 / 0.376 / 0.214 |
| P5011_MERSCOPE | 8.75% | 0.00% | 3.02% | 0.730 | 1.57% | 11,618 (0.198); 0.938 / 0.016 / 0.045 | 13,004 (0.222); 0.315 / 0.324 / 0.361 |
| P5011_XENIUM | 3.73% | 0.03% | 2.15% | 0.638 | 0.46% | 10,352 (0.091); 0.925 / 0.038 / 0.036 | 12,509 (0.109); 0.406 / 0.360 / 0.234 |

- **H1.** P7113 is 0.126 (CI 0.092–0.171; threshold 0.17) and P5011 is 0.227 (0.210–0.243; threshold 0.31). The plan's basis was 0.136 / 0.279.
- **H2.** P7113 is at 0.77% / 0.59%. **P5011_MERSCOPE is at 1.02%, above the 1% threshold** (§7 item 1); P5011_XENIUM is at 0.91%.
- **H3.** Agreement is 0.948 / 0.929 / 0.828 / 0.921, all at or above 0.80.
- **H7.** Every sample clears its threshold, by 0.09 to 0.26.
- **H8.** P5011_MERSCOPE is `broad_only` (A = 0.129) with the warning flag (coverage of segmented objects 0.106), as §5.4 expects. P7113 and P5011_XENIUM are `full`.
- **OD-B13.** The trigger is not met.

| Sample | Shared cells (share < 60 counts) | No second method | v1 without the below-60 rule | v1 (SEA-AD below 60) | v1 with LL below 60 | E2 LL rule alone | Below-60 SEA-AD cost | Below-60 LL cost | LL minus SEA-AD (trigger if > +0.08) |
|---|---|---|---|---|---|---|---|---|---|
| P7113_MERSCOPE | 30,000 (0.61) | 0.747 | 0.761 | 0.757 | 0.741 | 0.727 | 0.005 | 0.020 | -0.015 |
| P7113_XENIUM | 30,000 (0.71) | 0.637 | 0.648 | 0.646 | 0.572 | 0.561 | 0.002 | 0.076 | -0.074 |
| P5011_MERSCOPE | 30,000 (0.99) | 0.442 | 0.464 | 0.454 | 0.427 | 0.405 | 0.010 | 0.037 | -0.027 |
| P5011_XENIUM | 30,000 (0.77) | 0.543 | 0.556 | 0.552 | 0.535 | 0.522 | 0.004 | 0.021 | -0.016 |

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
| P7513 | <= 0.19 | 0.127 [0.088, 0.171] | 0.125 [0.088, 0.173] | 0.129 [0.091, 0.173] | 0.127 [0.092, 0.171] | 0.259 | – | 0.008 / 0.009 |
| P1212 | <= 0.19 | 0.125 [0.105, 0.155] | 0.163 [0.146, 0.188] | 0.140 [0.109, 0.180] | 0.199 [0.179, 0.225] | 0.312 | – | 0.010 / 0.012 |
| P7113 | <= 0.19 | 0.177 [0.136, 0.217] | 0.160 [0.125, 0.204] | 0.177 [0.134, 0.218] | 0.188 [0.150, 0.233] | 0.444 | – | 0.007 / 0.011 |
| P5011 | <= 0.33 | 0.320 [0.301, 0.340] | 0.323 [0.303, 0.353] | 0.337 [0.318, 0.357] | 0.342 [0.318, 0.366] | 0.294 | – | 0.033 / 0.017 |

| Sample | Table cells | Median counts | A (share >= 30 counts) | Gate level | Warning | Confident broad, table cells [H7 threshold] | Confident broad, segmented objects | Lineage | Supercluster | Implausible calls (H2 <= 1%) | WHB-SEA 7-class, >= 20 counts [H3 threshold] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P7513_MERSCOPE | 137,129 | 71 | 0.784 | full | no | 0.887 | 0.574 | 0.905 | 0.770 | 0.71% | 0.976 |
| P7513_XENIUM | 97,363 | 41 | 0.626 | full | no | 0.836 | 0.485 | 0.874 | 0.703 | 0.45% | 0.973 |
| P1212_MERSCOPE | 35,894 | 17 | 0.256 | broad_only | yes | 0.718 | 0.087 | 0.763 | 0.000 | 1.04% | 0.874 |
| P1212_XENIUM | 85,349 | 23 | 0.376 | full | no | 0.779 | 0.274 | 0.857 | 0.643 | 0.77% | 0.962 |
| P7113_MERSCOPE | 80,827 | 39 | 0.618 | full | no | 0.883 | 0.364 | 0.902 | 0.793 | 1.13% | 0.980 |
| P7113_XENIUM | 117,078 | 30 | 0.512 | full | no | 0.839 | 0.460 | 0.889 | 0.708 | 0.88% | 0.971 |
| P5011_MERSCOPE | 11,845 | 13 | 0.039 | broad_only | yes | 0.850 | 0.039 | 0.908 | 0.000 | 2.44% | 0.953 |
| P5011_XENIUM | 63,037 | 25 | 0.423 | full | no | 0.792 | 0.250 | 0.883 | 0.499 | 1.35% | 0.981 |

| Sample | COP argmax calls | Confident COP supercluster (H5 <= 2%) | Confident broad OPC (H5 <= 10%) | COP-derived share of confident OPC | COP suppressed (`flag_cop_suppressed`) | New vs legacy referee: disputes (share); new / legacy / tie (H10 >= 0.70) | WHB vs SEA-AD referee: disputes (share); WHB / SEA-AD / tie |
|---|---|---|---|---|---|---|---|
| P7513_MERSCOPE | 1.74% | 0.21% | 2.69% | 0.513 | 0.17% | 51,239 (0.374); 0.993 / 0.003 / 0.004 | 4,043 (0.029); 0.312 / 0.450 / 0.238 |
| P7513_XENIUM | 1.79% | 0.11% | 1.84% | 0.662 | 0.21% | 22,079 (0.227); 0.985 / 0.007 / 0.008 | 3,297 (0.034); 0.314 / 0.420 / 0.266 |
| P1212_MERSCOPE | 2.35% | 0.00% | 2.69% | 0.713 | 0.24% | 13,581 (0.378); 0.973 / 0.002 / 0.025 | 3,714 (0.103); 0.110 / 0.338 / 0.552 |
| P1212_XENIUM | 1.47% | 0.01% | 1.24% | 0.535 | 0.32% | 8,659 (0.101); 0.921 / 0.022 / 0.058 | 3,478 (0.041); 0.268 / 0.434 / 0.298 |
| P7113_MERSCOPE | 1.55% | 0.07% | 2.59% | 0.523 | 0.05% | 41,795 (0.517); 0.996 / 0.001 / 0.003 | 2,825 (0.035); 0.138 / 0.482 / 0.379 |
| P7113_XENIUM | 1.77% | 0.07% | 2.64% | 0.538 | 0.08% | 11,731 (0.100); 0.917 / 0.021 / 0.062 | 3,595 (0.031); 0.328 / 0.393 / 0.279 |
| P5011_MERSCOPE | 6.48% | 0.00% | 9.09% | 0.644 | 0.17% | 3,454 (0.292); 0.992 / 0.000 / 0.008 | 367 (0.031); 0.117 / 0.428 / 0.455 |
| P5011_XENIUM | 2.46% | 0.01% | 3.93% | 0.575 | 0.04% | 7,313 (0.116); 0.982 / 0.005 / 0.013 | 1,749 (0.028); 0.268 / 0.535 / 0.197 |

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
2. **OD-B13 is not triggered.** On all eight datasets the below-60 SEA-AD rule removes 0.2–1.0 points of confident broad coverage (v1 without the rule minus v1). Using the likelihood typer as the below-60 vote instead would remove 1.4–7.4 points more (`ll_minus_sea` −0.014 to −0.074). E2's LL rule alone, with no SEA-AD role, keeps 2.6–8.5 points less than v1. The LL part of M10 therefore stays in v1.1, to be recorded in the M3 PR. The separate LL value test (M3 item 6, OD-B8) is not part of this stage.
3. **H8 margin.** P1212_MERSCOPE's coverage of segmented objects (0.152) is 0.002 above the 0.15 warning threshold, and §5.4 expects no warning there. RESOLVE's coverage may move by up to ±0.05 (M4 exit), so P1212_MERSCOPE may acquire the warning flag. The warning never changes the gate level.
4. **P5011 reseg MERSCOPE lacks 10.8% of the panel.** The published `var` of 11,845 cells is `min_cells`-filtered to 268 features. M3 maps the present genes with a restricted lookup; plan §3.3 / §8.1 route more than 5% missing genes to a panel family of its own with a full PREP (M3b). Its reseg baselines are provisional until M3b re-measures them. P1212 reseg MERSCOPE lacks 1 gene (0.34%), within the 1% that a restricted lookup covers.
5. **Standalone runs on `min_cells`-filtered outputs.** `merxen annotate --from-clustered-h5ad` derives the panel from the published `var`. When a small sample lost a gene, the derived panel gets a new hash and the store has no bundle for it (P1212 and P5011 reseg failed this way). The declared panel has to be supplied through `merxen annotation-panel --panel-file` and `--panel-dir`. Adding `--panel-file` to `merxen annotate` is an open follow-up.
6. **Candidate tightenings** (not applied; for decision in the M3 PR):
   - **H1** (basis "soft + 0.03", now measured on production): P7513 ≤ 0.16, P7113 ≤ 0.16, P5011 ≤ 0.26 (P1212 stays at 0.17).
   - **H3:** P7113 ≥ 0.90 (baseline 0.948 / 0.929).
   - **H5:** confident COP supercluster ≤ 1% (maximum baseline 0.27%) and confident broad OPC ≤ 5% (3.0%).
   - **H7** (basis "E2 LL-rule coverage − 0.10"; now the SEA-rule baseline − 0.10 where that is stricter): P7513 X ≥ 0.50, P1212 X ≥ 0.41, P7113 X ≥ 0.55, P5011 M ≥ 0.36, P5011 X ≥ 0.46.
   - **H10:** ≥ 0.85 (minimum baseline on the flip-rule datasets 0.961).

   Resolvability (M3b) and RESOLVE (M4) can still lower coverage, so tightening H7 before M4 risks a failure that belongs to M3b.

## 8. Not measured in this stage

| Criterion | Where it is measured |
|---|---|
| H4 held-out-gene enrichment | M3 shadow item 4 (`scripts/acceptance/heldout_genes.py`) |
| H6 simulation precision of the v1 thresholds | the archived E2 HO simulation, scored at M8; H18 recomputes it in production |
| H9 canonical-marker plausibility (also part of H17) | M8 acceptance script |
| H12 cortical depth, H13 contracts, H15 reproducibility | M5 / M7 / M8 |
| H16 realised flag rates | M3 shadow item 5, M4 |
| H18 resolvability regression | M3b (production PREP) |
| MO1–MO11 (mouse) | M6, M6b, M9 |
| NP1–NP9 (gate P) | M13 |

M3 shadow items 2–7 (E8 segmentation comparison, X1 rescaling, held-out genes, flag rates, the LL value test and the glial JSD difference) are later stages of M3.

## 9. Registered thresholds (plan rev3 §14, verbatim)

The text below is §14 of `docs/plans/robust-celltype-annotation-plan.md` as of `06a04a1`. Section references (§n) point into the plan. The section title is shown in bold rather than as a heading; the text is otherwise unchanged.

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
| `metrics/jsd.csv` | Per pair × segmentation × kind (soft, argmax, confident, soft_ge30, setc_soft, setc_argmax) × region (whole_section, shared_mask): JSD, 95% CI, tiles, cells, unallocated mass |
| `metrics/compositions.csv` | The compositions behind `jsd.csv`, per platform (shares incl. unallocated, and renormalised 7-class shares) |
| `metrics/referee.csv` | E1 marker referee: WHB vs SEA-AD, new (v1 shadow) vs legacy, WHB argmax vs legacy, new vs SEA-AD |
| `metrics/second_vote_cost.csv` | OD-B13: coverage under the five second-vote variants on E2's 30k native cells, costs, trigger, E2's published LL-rule coverage |
| `metrics/exit_checks.csv` | M3 exit: production vs pilot agreement per sample, reference and subset |
| `metrics/runtime.csv` | MAP wall time, processor-seconds per 1k cells, peak RSS, missing panel genes per run |
| `runs/<pair>/<seg>/` | `merxen annotate` outputs: `map_manifest.json`, `panel/`, `<plat>/<sid>_mmc_<run>.parquet`, `<plat>/<sid>_ct_provisional.parquet`, `logs/`; for P1212 / P5011 reseg also `panel_varderived/` (the refused var-derived panel) and `inputs_view/` |
| `logs/`, `PROGRESS.txt` | Run logs with `/usr/bin/time` wall and max RSS; `metrics.log` |
