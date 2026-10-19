# HITL-0 runtime validation — 2026-10-19

Production code: `c2ad456`, based on `a9625a9`. This records a stock-model drift
check; it does not evaluate biological segmentation accuracy or a fine-tune.

## GPU tile check: passed

The optional `workflows/acceptance/hitl0_tile_check.nf` harness ran on Dwight,
under the existing `/tmp/merxen-dwight-gpu.lock`, using the existing production
environment. No active compute job was present at launch; desktop graphics
remained running. The task used read-only published inputs and wrote its
prediction, identity sidecar and report to a separate acceptance directory.

| Setting | Recorded value |
|---|---|
| Sample | P7513 Xenium |
| Native image | `morphology_focus`, DAPI then 18S |
| Source | Surviving source zarr used by the published run |
| Tile | Index 44 of the recorded global grid |
| Halo bbox, y0/y1/x0/x1 | 22272 / 28416 / 22272 / 28416 px |
| Grid tile size / overlap | 6144 / 256 px |
| Scoring bbox, y0/y1/x0/x1 | 23184 / 27504 / 23184 / 27504 px |
| Scoring rule | 656 px margin inside the tile core; exclude cut border labels |
| Model | `cpsam_v2`, float32 |
| Cellpose / torch | 4.2.1.1 / 2.14.0+cu130 |
| Flow / cellprob thresholds | 0.7 / -5.0 |
| Per-tile / final filters | Production settings; final area 5–400 µm² |
| Baseline cells / recomputed cells scored | 3745 / 3745 |
| TP / FP / FN at IoU 0.5 | 3745 / 0 / 0 |
| AP at IoU 0.5 | 1.000000 |
| Mean matched IoU | 0.9999953134 |
| Foreground IoU | 0.9999950559 |
| Acceptance requirement | AP and mean matched IoU both ≥ 0.99 |
| Peak torch GPU allocation | 6.374 GiB |
| Total task time, including input checksums | 173.5 seconds |

Checkpoint SHA-256:
`0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667`.

Published cell-mask SHA-256 was identical before and after:
`0a3e8b79ed2e8da1df6731c8fcd4be10a56e0cce241c89ae47f42cd193497b41`.
The published stitching-statistics checksum was also unchanged:
`bac2a39ebdbd2e374a0db5026f9c1ac51b50c6d27e129cf9475f79f1545666b4`.

The full report and prediction were preserved on Dwight under
`/srv/storage/MerXen/annotation_dev/hitl_acceptance/2026-10-19/hitl0-P7513-t44-rs5osM/`.
The Nextflow task work directory is
`/srv/storage/MerXen/tmp/hitl0-acceptance.rs5osM/work/15/529b75*/`.
The task work directory is temporary; the numerical record above remains in the
repository. Reproduction commands are in the
[testing guide](../hitl/testing-guide.md).

## Code checks: passed

The full Linux suite in Dwight's existing CI environment passed 761 tests, with
3 slow tests deselected. The final acceptance harness passed all 12 focused tests,
including actual Nextflow previews and output-path protection through symlinks.
Ruff lint/format checks and mypy passed. The full-suite environment was reused;
installation from the updated dependency lock remains covered by PR CI and the
container check below.

## Refreshed container: pending user-run check

The `becalia` account cannot access Dwight's Docker daemon, and `sudo -n docker`
reports that a password is required. Neither group membership nor Docker socket
permissions were changed. The user will run the container build and GPU smoke
test from their terminal using the commands in the testing guide. Operational
acceptance of the container remains pending that result.

## Confirmed HITL-1 behavior

- Preserve reuse of older stock outputs marked `unverified`; do not require
  their regeneration.
- Require matching verified provenance for custom-model reuse.
- Protect `/srv/storage/MerXen/results/` and its symlink aliases from custom-model
  writes; custom arms use separate output directories.
- Complete HITL-0 acceptance and merge before implementing HITL-1 from `main`.
