# Cellpose HITL implementation status

Implementation started on 2026-10-19 from `main` at `a9625a9`, on
`codex/hitl-segmentation`. The design remains the
[Cellpose HITL plan](cellpose-hitl-segmentation-plan.md). This first change is
limited to HITL-0; crop export, annotation, training and custom-model integration
are later milestones.

HITL-0 merged in [PR #74](https://github.com/bourdenxlab/MerXen/pull/74) at
`6e7481a`. Its fresh-lock GitHub CI passed 760 tests with 5 skips and 3 slow tests
deselected. The lock install uses `--no-deps` to preserve all generated pins
without re-resolving upstream unpinned Git URLs.

HITL-1 is implemented on `codex/hitl-1-model-plumbing`, based on that `main`.
The section below records its current validation state.

## Confirmed decisions

- Land HITL-0 before HITL-1.
- Use the current Dwight runtime as the baseline, after checking it directly.
- Align the container's CUDA minor with Dwight.
- Validate HITL-0 through Dwight's existing Conda path and defer container
  validation. The pipeline's container runtime remains Apptainer.

Dwight was checked on 2026-10-19: Cellpose 4.2.1.1, torch 2.14.0+cu130,
torchvision 0.29.0, CUDA 13.0, NVIDIA driver 580.126.09. Its cached `cpsam_v2`
checkpoint has SHA-256
`0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667`,
matching the design's expected weights. The new builder successfully loaded and
identified that checkpoint on CPU using Dwight's existing production environment.
No published masks were modified.

## HITL-0 implementation

- Pin Cellpose to 4.2.1.1 and bound torch to 2.14.x. The regenerated Linux lock
  uses torch 2.14.0 and torchvision 0.29.0, retaining unrelated git dependency
  revisions. Conda still resolves from `pyproject.toml`, so it can resolve a newer
  2.14 patch; every inference records the actual loaded torch patch/build.
- Explicitly select `cpsam_v2` for cells and independent DAPI-only nuclei, warn
  about ignored legacy `model_type`, and always pass the requested precision.
  The Python precision default now matches production's float32 setting.
- Validate the runtime and loaded checkpoint before replacing existing outputs;
  record successful inference identities beside masks and in statistics.
- Preserve sidecars on reuse. Mark older outputs `unverified` without inventing
  historical versions, weights or settings. Retain the original task log as the
  source to consult for historical evidence.
- Publish both sidecars, propagate their contents through enrichment and later
  segmentation registration, and keep model provenance separate from
  `writer_versions`.
- Use the CUDA 13.0 container base and expose NVIDIA devices to the nuclei process
  under the GPU profile.
- Keep legacy Nextflow Cellpose configuration text and all three segmentation
  process scripts unchanged, enforced by regression hashes. The environment hash
  changes, so environment rebuilds must be scheduled between pipeline runs.

## Validation and remaining acceptance work

Validation on Dwight: 761 tests passed in the full Linux suite (3 slow tests
deselected). The final acceptance-harness tests passed separately: 12 tests,
including real Nextflow previews of separate and protected output paths. These
checks used the existing CI environment; installation of the refreshed lock is
also checked by PR CI. Container validation is deferred. The local focused run
passed 175 tests with 6 platform-dependent skips. Ruff lint/format checks and
mypy pass. An isolated Nextflow preview accepts the named output channels and
existing ProSeg tuples.

The HD1 production-context GPU tile check passed on 2026-10-19 under the shared
GPU lock: all 3745 scored cells matched, AP at IoU 0.5 was 1.0, and mean matched
IoU was 0.9999953134. The published mask and statistics checksums were unchanged.
See the [validation record](../acceptance/hitl0-runtime-validation.md).

The user confirmed that the Conda path is the acceptance target for this
milestone. The Apptainer image build and GPU smoke test are deferred; the earlier
Docker/sudo test instructions were withdrawn. The updated Dockerfile remains an
unvalidated build recipe, and the site's existing CUDA 12.6 SIF has not been
refreshed for the new pins. The [testing guide](../hitl/testing-guide.md) contains
the repeatable Conda tile-check commands and the HITL-1 acceptance checklist.

The local macOS full-suite attempt crashed in native clustering code; validation
of the full suite therefore uses an isolated Linux checkout on Dwight.

The user confirmed that HITL-1 must retain legacy stock-mask reuse and protect
`$MERXEN_RESULTS_ROOT/`. HITL-1 follows acceptance and merge of HITL-0.

## HITL-1 implementation and validation

- Optional per-platform bundle/path/hash model selectors and threshold/filter
  overrides, including samplesheet inheritance and explicit zero values.
  Unset options leave the legacy config JSON unchanged.
- Strict `model.json` schema, actual weights checksums before/after loading,
  candidate opt-in, native image/channel order, full-affine pixel-size checks,
  exact runtime checks and a singleton evaluated tile size.
- Verified custom-cache identities include image, filtering and tiling context.
  All segmentation entry points refuse mismatches before replacement, including
  forced reruns and the combined pipeline's fast path. Downstream-only model
  selectors verify saved identities before scheduling work.
- Protected roots cover visible paths, symlink aliases and published links into
  old Nextflow work directories. Mutable source zarrs must also be independent;
  the read-only tile diagnostic can read the published source explicitly.
- Independent nuclei retain stock weights and global settings. C1–C5 and C7 are
  each marked once. The production loop calls `run_cellpose_tile`; pre-extraction
  hashes pin synthetic masks, logits and stitching statistics.
- A stock-only candidate fixture helper and executable instructions in the
  [testing guide](../hitl/testing-guide.md). No trained model or annotations are
  needed for these checks. The
  [model-selection reference](../stages/cellpose-hitl.md) documents the schema.

Local focused checks passed 217 tests with 10 platform/tool-dependent skips.
After the final incomplete-cache checks, all 125 segmentation tests passed
locally with 1 platform skip. Complete Linux CI passed 820 tests with 3 slow
tests deselected and no skips, including executable Nextflow checks. Results
are tracked in [PR #75](https://github.com/bourdenxlab/MerXen/pull/75).
A broader macOS run passed 739 tests with 18 skips and 3 slow tests deselected,
excluding the existing native clustering test module that crashes on this host.
Ruff and mypy pass. CI installs pinned Nextflow 26.10.2; Dwight's installed
26.04.4 also passed a main-workflow preview using the real HD1 source.

After the VPN connection was restored, the HITL-1 candidate-fixture GPU check
passed on Dwight using the production Conda runtime and shared GPU lock. All
3745 scored cells matched, AP was 1.0, mean matched IoU was 0.9999953134, and
the published mask/statistics checksums were unchanged. The verified identity
includes the candidate path and complete inference context. See the
[HITL-1 validation record](../acceptance/hitl1-model-validation.md) for the
report location and reproduction instructions. Runs starting after segmentation
verify saved mask identity when a model is selected, as confirmed by the user.
Container validation remains deferred.
