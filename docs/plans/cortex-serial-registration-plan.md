# Cortex serial-section registration for AT8 pair identification

> **Status:** revision 3, 2026-11-19. M0 (feasibility) is done. M1 is being built on the
> integration branch named below. Decisions D1-D11 are settled.

This plan extends [manual landmark registration](manual-registration-plan.md)
([stage docs](../stages/manual-registration.md)) to adjacent human cortex sections. The
reference is a Xenium section. The moving image is a laser-capture-microdissection (LCM)
section. Both are stained for AT8 (AF568), MAP2 (AF647) and DAPI and scanned on a Zeiss
Axioscan 7. The end goal is to find, for an AT8-positive tangle-bearing neuron in the Xenium
section, the same neuron in the LCM section, so that it can be cut out for proteomics. The cut
is drawn on the LCM section's own detected soma, in native LCM-scan coordinates. Registration
therefore only has to say which object is the partner, or that there is none.

Names follow the privacy rules in AGENTS.md. Pairs are aliases: `LCM1` and `LCM2`. A third pair
gets an alias once the user adds it to `.aliases.tsv`. Locations are `.env` variables.

## Decisions

| # | Decision |
|---|---|
| D1 | The job is to identify AT8 tangle pairs, not to register whole sections densely. A field is judged by how well it identifies partners (the pair excess over a shifted null), not by a whole-section error target. |
| D2 | M1 is built now, before the section-gap question is settled (user decision, 2026-11-17). Its robust field and landmark check are useful for region-level pairing and for every later milestone. |
| D3 | Cortex only. AT8 (AF568) is the target feature. MAP2 (AF647) carries texture and the radial (pia to white matter) direction. DAPI is demoted to one of three tissue-scale channels and is never used at nucleus scale. |
| D4 | The deformation model is a robust Student-t Gaussian process (GP) on landmarks and image evidence. The DAPI affine refinement and the TV-L1 flow are off for cortex pairs. |
| D5 | The landmark labelling schema stays at version 1. A check result is written as a JSON file and, for the store's own labelling, as a new root attribute of the prepared store. Only MerXen writes either. Both are keyed by pair id and tied to the labelling's content digest and the store's content digest. The viewer shows a check from the attribute or from a JSON file the user loads, and only while both digests match. |
| D6 | No numerical code is mirrored in the viewer before M4. The viewer shows what MerXen computed. |
| D7 | For cortex pairs, register writes the level-0 grid of the warped images (`materialize = full`), as the other presets do; a 5.5 µm preview (`materialize = preview`) is a setting for a quick look. Amended 2026-11-18: M1 first shipped the preview as the default, which is too coarse to match single cells (preset `cortex_serial_v2`). |
| D8 | The technician says the LCM1 and LCM2 sections were directly adjacent; M0 contradicts this. The user is preparing a new pair recorded as directly adjacent. M0b repeats the M0 tests on it after M1 and decides whether M2 (identity) is built. |
| D9 | The landmarks are labelled on another machine than the one that runs MerXen. The check therefore also runs on an uploaded labelling file and returns a JSON file that the viewer loads, so the user never copies a store's root attributes back and forth. |
| D10 | M1 leaves existing `same_section` and `serial_section` results valid: settings that M1 adds are left out of the provenance digest while they hold their defaults, so those runs are not re-registered. |
| D11 | The image evidence masks elongated vessel profiles, and a click on one gets noise along the vessel and is never kept at full weight (user decision, 2026-11-19): an obliquely cut vessel moves along its own axis between the sections while the tissue follows the field. The full mask is kept although it removes much of the evidence (LCM1 confirmations 74 to 42), because held-out errors on LCM2 fell (windows 46 to 37 µm, blocks 61 to 55 µm); the confirmed-offset bar of item 7 becomes 12 µm. The clicks-only field the classes use carries the same along-vessel noise, so a vessel-end click is not classed a misclick for an offset along its vessel. |

## What M0 found (2026-11-17)

M0 asked whether the two sections of a pair share cells, and so whether tangles can be paired at
all.

| Test | Same-section control (HD1) | LCM2 | LCM1 | Expected for touching 5 µm sections |
|---|---|---|---|---|
| Nuclei shared after local alignment | 78-93 % | at most 4-10 % | at most 3-6 % | 29-66 % |
| AT8 somata with a partner (pi) | - | about 0-5 % | about 2-8 % | 50-79 % |
| Image correlation at 1-3 µm scale (NCC) | about 0.95 | 0.03 (null level) | 0.03 (null level) | well above the null |

- As imaged, the sections share tissue-scale structure: vessel lumens, cracks, edges, and tangle
  density over about 1 mm (Spearman 0.66-0.85). Below about 10 µm they share almost nothing. The
  limit on pairing tangles is the tissue, not the registration.
- Ruled out as causes:
  - Registration error. The pair rate does not rise with a better field, and partners injected
    for 60 % of tangles give 10-30 times the observed excess.
  - Coordinate mistakes.
  - Weaker staining of the Xenium-processed section.
  - Focus.
- Forecast with the measured partner rate: about 0 confident pairs (posterior at least 0.9) on
  LCM2 and 1-3 on LCM1. If 60 % of tangles had partners, the same field would give about 180
  and 300.
- Registration accuracy, measured independently with LoFTR matches:
  - LCM2: 21-25 µm in well-matched areas, 48-53 µm elsewhere.
  - LCM1: 19-22 µm and 25-29 µm.
- Identification arithmetic: for one candidate 1 sigma from the prediction to reach a posterior
  of 0.9, at 6 tangles/mm², needs:
  - about 7 µm per axis when 3 % of tangles have partners;
  - about 28 µm per axis when 30 % do.

  In tangle-dense patches (about 100/mm²), identity is out of reach.

## Landmarks on LCM2 today

The user's LCM2 labelling (revision 2) has 71 pairs. 66 are enabled; 11, 12, 17, 18 and 20 are
disabled. The table gives leave-one-out (LOO) residuals of the 66 enabled pairs, in µm, from the
planning prototype of 2026-11-17.

| Model | LOO median | p90 | max | Flagged pairs |
|---|---|---|---|---|
| Similarity, no trimming (LOO of every enabled pair) | 108 | 300 | 492 | |
| Similarity with trimming (the viewer's live fit; register's landmark stage) | 73 | 391 | 565 | 13 trimmed: 13, 16, 51, 57-62, 64-66, 69 (median 58 over the 53 kept) |
| Affine | 121 | 240 | 400 | |
| Quadratic polynomial | 62 | 109 | 313 | |
| Robust GP, clicks only (one Matérn component) | 42 | 99 | 292 | p < 0.01: 3, 13, 19, 60, 62, 71 (13 and 71 survive Bonferroni) |
| Robust GP, clicks plus M0's tissue windows (two components) | 34-35 | 97-111 | 183-186 | |

- The trimmed similarity's flags fall mostly in clusters near the corners of the section
  (x 21-22.5 mm, y 4-7 mm; x 1.5-2.3 mm, y 3.2 mm). A similarity cannot follow the bending
  there. The robust GP fits 10 of the 13 flagged pairs with p ≥ 0.01 (pair 62 sits at 0.010), to
  within 20-65 µm.
- Pair 13 is about 145 µm off the smooth fit, on a pial-edge corner. One image check contradicted
  it, but an independent re-check found no image support for the click, the prediction or any
  alternative: inconclusive. Re-clicking it on a nearby point feature is a precaution.
- Pair 71 sits on the right vessel but about 290 µm along it from the predicted point. Image
  windows nearby show a local deviation of about 180 µm in the same direction, so part of the
  offset is real local deformation and part is the click's position along an oblique vessel.
- Pairs 62, 67 and 68 lie in grey matter with almost no image windows. The image confirms all
  three clicks (to 3-21 µm), yet the robust fit down-weighted them and missed them by about
  110 µm. There the field was wrong, not the clicks. M1 therefore keeps strongly
  image-confirmed clicks at full weight (amendments A1-A3 in the M1 specs), and grey-matter
  gates in M2 need a larger nugget (about 40 µm per axis against 25 µm elsewhere).
- The disabled pairs 11, 12, 18 and 20 are 1.0-1.3 mm off the enabled fit. Fitted together with
  all 71 pairs, they get weights of 0.004-0.007. Pair 17 fits (82 µm off, p 0.07).
- The image check of the prototype confirmed 32 of the 66 clicks, among them pairs 2 and 3, which
  geometry alone down-weights, and pairs 60 and 62, which it flags.
- The landmark scatter, about 30-40 µm per axis, is not click imprecision. Vessels in
  perivascular channels and tissue edges move by tens of µm between adjacent sections. Round
  parenchymal vessel profiles and bifurcations make better landmarks than vessels floating in
  sulci or points on long straight edges. Along such structures the scatter is about 60 µm;
  across them it is about 27 µm, against about 40 µm for a clean click.

## Pipeline after M1

```
[1] prepare (unchanged): CZI -> OME-TIFF pyramids -> prepared store
[2] viewer (user's machine): landmark pairs; the live similarity fit is labelled "rigid"
[3] upload the store's zarr.json (it holds the labelling) to the host
[4] merxen manual-registration-check --labelling <uploaded zarr.json>   (on the host)
      robust GP of the clicks
      -> image check of every click (AT8+MAP2+DAPI band-pass 20 µm and MAP2 orientation,
         2.76 µm/px)
      -> optional null-calibrated tissue-scale windows
      -> robust GP of clicks + windows
      -> per pair: class, LOO, p, weight, verdict; suggestions for the next clicks
      -> <dataset>_<preset>_check.json and .png
      (run without --labelling on the host's own store, it also writes the check attribute)
[5] download the check JSON; viewer: Load check file. The landmark table shows the check while
    it matches the labelling in the viewer
[6] manual_registration_register.nf, preset cortex_serial (on the host, after import)
      similarity guards -> the same robust field -> pull flow on an 11 µm grid
      -> level-0 store (or a 5.5 µm preview, or none) -> QC per channel role
      -> field JSON, flow, summary
[7] viewer: the result store, with the landmarks pushed through the field
```

The field enters register as a pull flow, in the format of the existing TV-L1 flow
(`flow_yx` in working pixels plus `working_to_reference`). Landmark push, tissue QC, the result
CSVs and the viewer overlays therefore work unchanged. The registration core used by
`REGISTER_IMAGES` is not modified.

The viewer features of steps [2], [5] and [7] need a viewer built from the viewer's integration
branch; no release contains them before M5. The orchestrator builds test installers of that
branch with the viewer's packaging workflow (a manual run builds every platform without
publishing) and gives the user the macOS disk image. A viewer installed with pip can instead be
installed from the branch.

## Milestones

**M0. Facts and feasibility.** Done; results above. The evidence is archived under
`$MERXEN_EVIDENCE_ROOT/manual_registration/cortex_serial/m0/`.

**M1. Robust field, landmark check, fast register** (this plan's work packages, about 2.5
weeks).
- Contents:
  - the robust Student-t GP field (`landmark_field.py`);
  - image evidence (`image_evidence.py`); the window NCC moves there from the test-landmark
    script;
  - the `cortex_serial` preset;
  - `merxen manual-registration-check`: on the store's labelling or an uploaded labelling file,
    with a JSON report and, for the store's own labelling, a prepared-store attribute;
  - the field as a pull flow;
  - `materialize = none | preview | full`;
  - QC per channel role;
  - Nextflow wiring and docs;
  - viewer display of the check (from the attribute or a loaded check file) and of preview
    result stores;
  - a synthetic phantom and golden vectors.
- Acceptance. These run on real data on the host; numbers are for the LCM2 revision-2 labelling
  unless stated.
  1. Robust field, clicks only: LOO median ≤ 45 µm and p90 ≤ 105 µm. The untrimmed similarity
     gives 108 and 300. The earlier bar of ≤ 65 µm, set on 22 pairs, stays a floor for future
     labellings.
  2. With tissue windows: click LOO median ≤ 38 µm.
  3. Clicks only: pairs 13 and 71 have p < 0.01, and pair 71 has a weight below 0.15. With the
     default settings (windows on), pairs 13 and 71 are classed `likely_misclick` or `check`.
  4. Pairs 2 and 3 are never classed `likely_misclick`.
  5. At least 9 of the 13 pairs that the trimmed similarity flags are classed `ok` or
     `kept_confirmed`.
  6. With all 71 pairs fitted, pairs 11, 12, 18 and 20 get weights below 0.05 and pair 17 stays
     above 0.5.
  7. Image check:
     - none of pairs 1, 2, 7, 8, 9, 14 and 23 is contradicted, and at least 4 of them have a
       confident fused estimate within 25 µm of the click (revised after the first M1
       builds: the earlier wording came from a different method, and an edge click is
       rarely confirmed outright);
     - the median image offset of confirmed clicks is ≤ 12 µm (revised with D11 from 10 µm:
       with vessels masked, the confirmations that remain are tissue matches, 10.2 µm on
       LCM2 against 8.4 µm when vessels could confirm themselves);
     - at least 20 % of enabled clicks get a verdict (confirmed or contradicted).
  8. Register with `cortex_serial` and a preview:
     - with the defaults (image check and tissue windows): ≤ 30 min and ≤ 9.9 GB peak memory;
     - without tissue windows: ≤ 12 min and ≤ 8.9 GB;
     - without tissue windows and without the image check (landmarks only): ≤ 6 min and
       ≤ 3.2 GB;
     - (the memory limits are the measured in-process peaks plus 25 %, revised after the
       first M1 builds; the original 10, 8 and 2.5 GB were estimates. They are not measured
       on the `merxen` command line, which still imports PyTorch and Cellpose;)
     - the result store is at most 1 GB and opens in a viewer built from the integration branch
       with the landmarks at their final positions.
  9. Synthetic phantom:
     - on a phantom that matches the model, 30-48 % of grid errors fall inside 1 sigma and
       80-92 % inside 2 sigma (a Gaussian gives 39 % and 86 %), on grid nodes at least 2.5 mm
       apart, pooled over 10 seeds;
     - every gross (1 mm) misclick gets a weight below 0.05 and p < 1e-6;
     - at most 3 % of clean clicks are falsely flagged on the matching phantom, and at most 6 %
       on the realistic one (ridge clicks and a second length scale).
  10. LCM1 (16 derived landmarks, revision-1 snapshot): LOO median ≤ 25 µm, and pair 16 is
      flagged.
  11. The check of the uploaded LCM2 labelling file gives the same pairs, field and suggestions
      as the check of the host's store with the same labelling.
- Useful on its own:
  - the best field the landmarks and images support;
  - misclick triage while clicking, on the user's machine;
  - suggested click locations;
  - preview registrations in minutes instead of an hour.

**M0b. Test on an adjacent pair** (after M1, about 3 days). The M0 tests run unchanged, with
the M1 field, on the new pair recorded as directly adjacent:
- shared nuclei;
- the AT8 partner rate, with injection controls;
- fine-scale NCC.

Gate: M2 is built only if one of these holds:
- the partner rate pi is at least about 0.3 on that pair;
- the user wants region-level pairing (tangle-bearing neurons from the same roughly 1 mm area and
  layer).

If the new pair also shares no cells, the protocol is investigated before any identity work:
ribbon order, and material lost or not imaged at the facing surfaces.

**M2. AT8 targets, scoring with "no partner", review, export** (about 2-2.5 weeks, both
repositories).
- AT8 census: tiers A, B and lenient, level-0 outlines, a saturation flag, grey-matter targets
  at least 50 µm from edges. A one-time annotation of a few 1 mm² fields measures recall.
- Candidates: moving tier A/B somata inside the 99 % gate (at most 200 µm away). The posterior
  is P_i = pi LR_i / ((1 - pi) + pi Σ_j LR_j), with LR_i = N(x_i; q, Σ) / λ_local.
  - pi is measured per pair.
  - Σ is the GP posterior covariance plus a local nugget of 30 µm per axis until measured.
  - Core size is the only morphology term.
- Shifted-null false-call tables.
- A viewer review panel: native LCM crop through the local affine, numbered candidates,
  P(no partner), and keys to confirm, reject, defer or mark no partner.
- Export: GeoJSON/CSV of the partner outlines in native LCM-scan px and µm.

Acceptance:
- ranked lists and false-call tables for both pairs;
- the top 50 per pair reviewed, with the time per item and blind-subset agreement recorded;
- outlines overlaid on the native LCM scan.

**Gate for M3.** M3 is built only if pi is not negligible and many candidates sit at posteriors
of 0.3-0.8, that is, limited by sigma.

**M3. Local evidence at targets** (about 2 weeks).
- LoFTR matches binned per 500 µm.
- 800 and 1200 µm windows centred on each target, with ambiguity tests.
- A cross-source hold-out.
- A seam test of the reference scans (their stitching grid survives band-pass filtering).

Acceptance: the pair excess within 15-25 µm rises by more than its SD compared with M2.

**M4. Live clicking loop** (about 1.5 weeks; viewer; optional). The viewer mirrors only
`landmark_field.py`, pinned by its golden vectors, to give:
- live LOO and SD maps;
- next-worst-pair navigation;
- suggestions shown as an ellipse, without a pre-placed moving point.

Built only if clicking is the bottleneck.

**M5. Feedback, validation, docs, merge** (about 1.5 weeks).
- Confirmed pairs feed back into the field with 15 µm noise, tightened to about 8 µm only when
  corroborated.
- A validation report on LCM1, LCM2 and the new pair.
- Stage docs.
- The integration branches merge into `main` with the user's approval, and the viewer is
  released.

## M1 work packages

| WP | Repo | Scope | Depends on |
|---|---|---|---|
| WP0 | MerXen | This plan; the integration branch added to `INTEGRATION_BRANCHES` (PR into `main`) | - |
| WP1 | MerXen | `landmark_field.py`: robust Student-t GP (affine mean, one Matérn 5/2 scale for clicks alone and two with windows, marginal-likelihood hyperparameters, EM weights that divide each observation's 2x2 noise, a fixed per-observation noise term for windows), closed-form LOO with chi-square p-values, posterior mean and SD, target-weighted greedy suggestions, serialisation; the phantom in `simulate.py`; golden vectors | WP0 |
| WP2 | MerXen | `image_evidence.py`: channel roles, band-pass and orientation representations at 2.76 µm/px, empirical nulls, the per-click check, tissue-scale window observations; the window NCC moved from `scripts/manual_registration/derive_test_landmarks.py` | WP0 |
| WP3a | MerXen | The landmark check: settings, the `cortex_serial` preset, channel roles, the field fit and classes, `merxen manual-registration-check` (store or uploaded labelling) with its JSON and attribute, the Nextflow check workflow, docs, CHANGELOG | WP1, WP2 |
| WP3b | MerXen | Register through the field: the pull flow, `materialize`, per-role QC, the field outputs and CSV columns, Nextflow register wiring, docs, CHANGELOG | WP3a |
| WP4 | viewer | Check columns from the attribute or a loaded check file, rigid-fit labels, placement of preview result stores | WP0 (contracts) |

WP1, WP2 and WP4 run in parallel. WP3a starts when WP1 and WP2 are merged, and WP3b when WP3a is
merged; the check reaches the user before the register path. M1 closes when the real-data
acceptance passes.

## Branching

- Integration branch `feature/cortex-serial-registration` in MerXen, cut from `main` after this
  plan is merged. The viewer repository gets a branch of the same name, cut from its `main`
  (v0.5.2).
- Work-package branches `feature/cortex-serial-registration-wp<N>-<slug>` merge into it by PR
  after CI and review. Keep the integration branch current by merging `main` into it; never
  rebase it.
- The branch is listed in `INTEGRATION_BRANCHES` of `scripts/ci/select_tests.py`, so the PR that
  merges it into `main` runs the full suite. It reaches `main` only with the user's approval and
  is deleted when the effort closes.
- Commit prefixes follow AGENTS.md. A user-visible change adds its CHANGELOG entry under
  [Unreleased] in the same PR, in both repositories. PR titles and bodies go through
  `python3 scripts/privacy/check_private_text.py --stdin`.

## Interfaces added in M1

- **Check report and attribute.** Schema `merxen.manual_registration.check`, version 1.
  - `merxen manual-registration-check` always writes `<dataset>_<preset>_check.json`: the check
    payload plus one extra key, `details`, which readers ignore. The viewer loads this file.
  - On the store's own labelling it also writes the payload as the prepared-store root attr
    `merxen_manual_registration_check`, under the store's write lock. It never writes the attr
    for an uploaded labelling file. Prepare and the viewer never write it, and prepare does not
    carry it over a rebuild.
  - Per pair, keyed by id: enabled, used, weight, LOO (µm), p-value, the clicks-only weight, LOO
    and p-value, rigid LOO, rigid flag, image verdict, image offset and snap position, class.
    Also the field summary, the suggestions, and the labelling content digest and store content
    digest it belongs to.
  - The viewer shows a check only while both digests match its labelling and store.
- **Field model.** `<run>_landmark_field.json` (schema `merxen.manual_registration.landmark_field`,
  version 1): the observations, hyperparameters and weights from which the field is recomputed
  exactly, with a digest.
- **Field flow.** `<run>_field_flow.npz`: the field as a pull flow, in the format of the TV-L1
  flow, plus the posterior SD map.
- **Result-store materialization.** `images.<role>.materialization` (`mode`, `level_factor`,
  `pixel_size_um`). A preview store holds a pyramid level of the reference crop and the moving
  image warped onto it. Viewers released before the WP4 change place such a store wrongly.
- **Channel roles.** Samplesheet column `channel_roles`, for example
  `at8=AF568;map2=AF647;dapi=DAPI`, with optional `reference:` / `moving:` prefixes. Required by
  `cortex_serial`. On a command line the value is quoted, because the shell splits at `;`.
- **Classes and verdicts.**
  - Classes: `ok`, `kept_confirmed` (a geometric outlier that the image confirms: real
    deformation or a structure offset, kept), `check`, `likely_misclick`, `excluded`, `disabled`.
  - Verdicts: `confirmed`, `contradicted`, `ridge`, `inconclusive`, `not_checked`.
  - A pair is judged on the lower p-value and weight of the clicks-only field and the final
    field, so tissue windows cannot hide a misclick.
  - Nothing is disabled or deleted automatically. A ridge verdict turns the click into an
    along/across constraint: 1.49 and 0.67 times the click noise (60/27 µm at the measured
    40 µm click scatter).

## Validation

1. **Synthetic phantom** (`simulate.py`). The phantom has:
   - a known two-scale field;
   - structure offsets, anisotropic along ridges;
   - gross (1 mm) and medium (30-150 µm) misclicks;
   - window observations with anisotropic noise, some of them wrong.

   Checks: SD calibration, misclick weights, false flags, and the gain from windows.
2. **Golden vectors.** They pin `landmark_field.py` (fixed-hyperparameter outputs, the fitted
   state's objective, predictions and LOO, suggestions) for the M4 mirror.
3. **Real data.** Host only, `pytest -m slow`, never in CI.
   - LCM2 and LCM1 run from snapshots of the labellings, under
     `$MERXEN_EVIDENCE_ROOT/manual_registration/cortex_serial/snapshots/`. The live stores keep
     changing while the user clicks.
   - Register runs read the prepared stores, take the labelling from the snapshot, and write
     only to temporary result folders.
4. **End-goal measure, from M0b on.** The AT8 pair excess over a 150-300 µm shifted null within
   10, 15, 25 and 50 µm, per field and pair. LCM2 baseline: 4.6 ± 2.0 pairs within 10 µm, and
   25 ± 10 within 50 µm.

## Not worth doing

- Nucleus-scale DAPI refinement, TV-L1 and cell-band flow on these pairs.
- CPD on soma sets as a driver of the warp.
- A dense GMRF with sulcal cuts.
- GPU intensity refinement.
- Fusing seven representations.
- Self-confirming gap-fill passes.
- Uncorroborated 4-5 µm soma anchors.
- A labelling schema bump.
- Several mirrored numerical modules.
- New heavy dependencies.

## Risks

- **Few partners (pi of 1-6 %).** Perhaps a few confident pairs per section pair. M0b decides.
- **Interior coverage.** LCM2-like interiors stay at 30-60 µm. Window coverage there is
  11-28 %, against about 50 % near edges.
- **Structure offsets versus real sub-millimetre strain** are not separable from clicks alone.
  The image check is the arbiter for single clicks.
- **Reference stitching seams** survive band-pass filtering and may bias windows. Flat-field
  correction is deferred to M3.
- **AT8 detection recall** is unmeasured until M2. About 14 % of strong LCM tangles are clipped;
  a rescan at a shorter AT8 exposure would help.
- **Little data.** Tuning rests on two pairs, and LCM1's landmarks were derived from DAPI
  windows.
- **Viewer builds.** The check display, the check-file loop and preview result stores need a
  viewer built from the integration branch until M5 releases it. Released viewers place preview
  stores wrongly.
- **Reused results keep their files.** A `same_section` or `serial_section` result reused after
  the upgrade keeps the CSVs and summary it was written with, without the new columns; readers
  treat the missing columns and keys as a similarity run.

## Questions open with the user

1. For the new pair: the ribbon or section numbers and the cutting order, and whether anything
   was discarded between the two sections.
2. Can the LCM slides be rescanned at a shorter AT8 exposure?
3. What counts as a usable partner (filled soma only, cap, process)? What posterior is
   acceptable? Is a tangle-bearing neuron from the same small region and layer an acceptable
   fallback?
4. Is a bench control possible: two sections from a spare block, recorded as consecutive and
   stained and scanned the same way?
5. How is the viewer installed on the labelling machine: the packaged macOS app or a pip
   environment? This decides how the user gets the integration-branch build.
