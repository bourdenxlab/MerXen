# Cellpose HITL implementation status

Implementation started on 2026-10-19 from `main` at `a9625a9`, on
`codex/hitl-segmentation`. The design remains the
[Cellpose HITL plan](cellpose-hitl-segmentation-plan.md). This first change is
limited to HITL-0; crop export, annotation, training and custom-model integration
are later milestones.

## Confirmed decisions

- Land HITL-0 before HITL-1.
- Use the current Dwight runtime as the baseline, after checking it directly.
- Align the container's CUDA minor with Dwight.

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
also checked by PR CI and the pending container build. The local focused run
passed 175 tests with 6 platform-dependent skips. Ruff lint/format checks and
mypy pass. An isolated Nextflow preview accepts the named output channels and
existing ProSeg tuples.

The P7513 production-context GPU tile check passed on 2026-10-19 under the shared
GPU lock: all 3745 scored cells matched, AP at IoU 0.5 was 1.0, and mean matched
IoU was 0.9999953134. The published mask and statistics checksums were unchanged.
See the [validation record](../acceptance/hitl0-runtime-validation.md).

The refreshed container still needs its build and CUDA smoke test. Docker access
is unavailable to the current Dwight account, and passwordless sudo is disabled.
The [testing guide](../hitl/testing-guide.md) contains the exact build, GPU smoke
and repeatable tile-check commands, plus the HITL-1 acceptance checklist.

The local macOS full-suite attempt crashed in native clustering code; validation
of the full suite therefore uses an isolated Linux checkout on Dwight.

The user confirmed that HITL-1 must retain legacy stock-mask reuse and protect
`/srv/storage/MerXen/results/`. HITL-1 follows acceptance and merge of HITL-0.
