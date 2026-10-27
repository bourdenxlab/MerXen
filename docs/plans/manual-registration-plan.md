# Manual landmark registration

> **Status:** revision 1, 2026-10-27. Design complete; implementation starts on the integration branch
> named below. Decisions D1-D4 are settled (table below).

Landmark-based registration of a moving image onto a reference image, with the napari viewer
(`napari-spatialomics-viewer`) as the labelling tool. It replaces the Xenium Explorer rigid alignment
that `REGISTER_IMAGES` needs today, and feeds the same guarded affine refinement and TV-L1 optical flow.

Typical pairs: a laser-capture-microdissection (LCM) section onto an adjacent Xenium section, both
scanned on a Zeiss Axioscan 7; or a post-Xenium IF scan onto its Xenium morphology image.

Names and paths follow the privacy rule in CLAUDE.md: samples are aliases (`LCM1`, `LCM2` for the two
Zeiss CZI pairs; `HD1` for the post-Xenium IF ground-truth pair), and locations are `.env` variables.

## Decisions

| # | Decision | Status |
|---|---|---|
| D1 | CZI reader. `czifile` cannot join the base lock: squidpy caps `imagecodecs<2026`, and current `czifile` needs `imagecodecs>=2026.3.6` and Python 3.12+. | Decided: an isolated `czi-runtime` environment (`requirements/requirements.czi.lock`, `envs/environment.czi.yml`). Prepare calls it through a subprocess bridge (`MERXEN_CZI_PYTHON`, Nextflow `manual_registration_czi_python`). The base lock, `envs/environment.yml` and the main pipeline's `-resume` are untouched. |
| D2 | Aliases for the two Zeiss CZI pairs. | Decided: `LCM1`, `LCM2` (private `.aliases.tsv`). Reference = the Xenium section, moving = the LCM section. Validation uses `LCM1`. |
| D3 | `serial_section` preset values, and the CZI acceptance gate (no ground truth). | Decided: v1 values below are provisional, with one tuning round expected after the first comparison. Gate: refinement and flow must not worsen tissue-level DAPI correlation or held-out landmark residuals relative to the landmark fit. `same_section` results on the serial pair are reported, not gated. |
| D4 | Base of the integration branch. | Resolved: the optic-flow residual passes are on `main` (PR #90), so the branch is cut from `main`. |

## Goal and workflow

```
samplesheet row: id (alias), reference image, moving image, optional scenes / settings, presets
  [1] nextflow run workflows/manual_registration_prepare.nf
        CZI -> pyramidal OME-TIFF (or validate an OME-TIFF) -> image-only SpatialData "prepared store"
        VIEWER_CACHE (image-only mode) -> pyramids for both images
  [2] viewer: Dataset loader > Manual Registration > Load image data
        optional crop per image -> coarse rotate / flip / shift of the moving image -> landmark pairs
        -> Finish labelling (>= 8 pairs) writes one root attrs key of the prepared store
  [3] nextflow run workflows/manual_registration_register.nf
        per (row x preset): crop -> robust similarity fit from the landmarks -> guarded DAPI affine
        refinement -> guarded TV-L1 flow -> warp every moving channel onto the reference grid
        compare presets per id
  [4] viewer: the same button on a result store shows the reference, the warped moving image and the
        landmarks pushed through the final mapping
```

Neither workflow is a stage of `workflows/main.nf`. The register workflow writes the final matrix as a
3x3 CSV in the Xenium Explorer convention (moving level-0 px -> reference level-0 px, full uncropped
frames), so it can be used as `xenium_registered_images_csv` in the main pipeline.

Inputs: OME-TIFF or Zeiss `.czi`. CZI channel names are read from the file metadata; a file with several
scenes needs a `scene` column; Z is max-projected. The reference keeps only the channels the moving
image also has; the moving image keeps and warps all of its channels. DAPI is the registration channel
and the default landmark channel.

## Branching

- Integration branch `feature/manual-registration`, cut from `main`. Work-package branches
  `feature/manual-registration-<wp>-<slug>` (for example `feature/manual-registration-mx1-contracts`)
  merge into it by PR after CI and review. Keep it current by merging `main` into it; never rebase it.
  It reaches `main` only with the user's approval, and is deleted when the effort closes.
- Viewer repository: `feature/manual-registration` with work-package branches
  `feature/manual-registration-<wp>`; viewer version 0.4.0.
- Commit prefixes per Agents.md. PR titles and bodies go through
  `python3 scripts/privacy/check_private_text.py --stdin`.

## Conventions (shared by both repositories)

- A point is `(x, y)` (column, row) in level-0 pixels of the full, uncropped image it belongs to.
  Integer values are pixel centres.
- Every matrix maps moving -> reference (`p_ref = M p_mov`, 3x3), the Xenium Explorer convention.
- Coarse transform (moving image only): `rotation_deg` in (-180, 180], `mirror_x`, and `offset_um`
  (displayed moving centre relative to the reference centre, reference µm).
  `M_coarse = T(c_r + o/p_r) R(theta) F(mirror_x) S(p_m/p_r) T(-c_m)`. Positive angles are
  counter-clockwise on screen.
- Crops: integer half-open rectangles per image in its own full pixel frame. The workflow snaps them
  inward to multiples of 64 px, so every pyramid level with a factor dividing 64 crops exactly. An
  effective side under 1024 px is an error. Matrices convert between frames with
  `M_crop = T(-o_r) M_full T(o_m)`.
- Root attrs keys of the stores, each with a single writer:
  `merxen_manual_registration_store` (prepare / register), `merxen_manual_registration` (the viewer
  only), `merxen_manual_registration_result` (register).

## Work packages

| WP | Repo | Scope | Depends on |
|---|---|---|---|
| MX1 | MerXen | Contracts: shared registration settings split out of `ImageRegistrationConfig`, image-only `ViewerCacheConfig`, manual-registration config models, schema, geometry, landmark fit, channel names, golden vectors | - |
| MX2 | MerXen | Reusable registration core extracted from `REGISTER_IMAGES` (output byte-identical), tunable nonrigid length scales, point push/pull | MX1 |
| MX3 | MerXen | CZI reader and converter, pyramidal OME-TIFF writer, `merxen convert-czi`, isolated CZI runtime | MX1 |
| MX4 | MerXen | Prepare: prepared-store builder, image-only viewer caches | MX1, MX3 |
| MX5 | MerXen | Presets, cropped sources, landmark fit, register run, QC, compare, synthetic pairs | MX1, MX2, MX4 |
| MX6 | MerXen | Nextflow workflows, CLI registration, docs, `.env` variables, HD1 and CZI validation | MX3, MX4, MX5 |
| VW1 | viewer | Landmark model, store I/O, Manual registration widget | MX1 (golden file) |
| VW2 | viewer | Registration controller (layers, crops, coarse transform, landmark mode, saving, result mode) | VW1 |
| VW3 | viewer | Hooks in `viewer.py`, Dataset loader section, new tab, docs, packaging | VW1, VW2 |

Merge order: MX1 -> (MX2, MX3, VW1) -> MX4 -> MX5 -> MX6; VW1 -> VW2 -> VW3.

## Presets

`same_section` is today's `REGISTER_IMAGES` defaults with `nonrigid_enabled = true` (nucleus scale).
`serial_section` works at tissue scale (layers, vessels, white-matter boundaries) for adjacent sections
whose nuclei do not match one to one. Every field can be overridden per row and through params. The
preset name is part of every run name and output path, so both presets can run on one pair and be
compared.

| Field | same_section | serial_section |
|---|---|---|
| `matrix_scale_tolerance` | 0.02 | 0.05 |
| `registration_pixel_size_um` | 0.85 | 6.0 |
| `refinement_window_um` / `min_refinement_windows` | 200 / 20 | 1200 / 12 |
| `max_refinement_shift_um` / `_rotation_deg` / `_scale_change` | 50 / 2 / 0.02 | 200 / 5 / 0.05 |
| `refinement_min_window_quality` / `refinement_require_moving_tissue` | 0.3 / false | 0.2 / true |
| `nonrigid_pixel_size_um` | 3.4 | 12.0 |
| `nonrigid_smoothing_sigma_um` | 10 | 100 |
| `nonrigid_max_displacement_um` / `_max_cycle_error_um` | 20 / 5 | 200 / 30 |
| `nonrigid_min_local_correlation` | 0.2 | 0.15 |
| `nonrigid_min_jacobian` / `_max_jacobian` | 0.5 / 2.0 | 0.67 / 1.5 |
| `nonrigid_highpass_sigma_um` / `_local_correlation_sigma_um` | 25 / 15 | 120 / 80 |
| `nonrigid_local_residual_tiles` | true | false |
| `landmark_outlier_floor_um` | 10 | 30 |

## Golden vectors

`tests/test_data/manual_registration/golden_v1.json` is the cross-repository contract for the coarse
matrix, the robust similarity fit, leave-one-out residuals and pair colours. It is generated by
`scripts/manual_registration/make_golden_vectors.py` and copied byte for byte into the viewer; both
repositories pin its SHA-256.

## Validation gates

- Unit, integration and Nextflow stub tests in CI, including a synthetic end-to-end pair with a known
  warp.
- HD1 post-Xenium IF -> Xenium (ground truth): landmarks simulated from its Explorer matrix; held-out
  error is measured against the noise-free mapping.
- `LCM1` Zeiss CZI pair (no ground truth): both presets run and are compared on landmark residuals and
  tissue-level DAPI correlation before and after refinement and flow (gate D3).
- The test-data location lives only in `.env` (`MERXEN_MANUAL_REGISTRATION_TEST_DIR`).
