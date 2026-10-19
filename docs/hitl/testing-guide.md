# Testing the Cellpose HITL changes

Start with HITL-0's runtime and provenance checks. HITL-1 follows after HITL-0's
acceptance and merge. The published baseline is `/srv/storage/MerXen/results/`;
all acceptance predictions and reports must go outside that directory.

## 1. Check the code without running segmentation

On Dwight, check out the implementation branch in a separate test checkout:

```bash
cd /home/becalia/code/MerXen
git fetch origin
export MERXEN_TEST_CHECKOUT=$(mktemp -d /srv/storage/MerXen/tmp/hitl-checkout.XXXXXX)
git worktree add --detach "$MERXEN_TEST_CHECKOUT" origin/codex/hitl-segmentation
cd "$MERXEN_TEST_CHECKOUT"
export PYTHONPATH="$MERXEN_TEST_CHECKOUT/src"
/srv/storage/MerXen/ci-venv/bin/python -m pytest \
  tests/test_segmentation/test_model_identity.py \
  tests/test_segmentation/test_output_paths.py \
  tests/test_scripts/test_hitl0_tile_check.py \
  tests/test_workflows/test_segmentation_subworkflow.py \
  tests/test_workflows/test_env_lock_sync.py
```

Expected: every selected test passes. These use small synthetic inputs and
mocked model loading, so this step does not run Cellpose on your images or
change the production environment. It checks explicit model selection,
checksums, legacy reuse, provenance through staged symlinks and workflow wiring.

## 2. Check one real P7513 tile

Use the production environment checked on 2026-10-19, rather than rebuilding the
environment used by active pipeline runs. If that cached environment has since
been removed, select another environment satisfying the new pins first.

From the test checkout above:

```bash
export MERXEN_TEST_PYTHON=/srv/storage/MerXen/work/conda/env-0e319c40ef50b2834c65ae2f7f53bbcd/bin/python
export MERXEN_CHECK_ROOT=$(mktemp -d /srv/storage/MerXen/tmp/hitl0-check.XXXXXX)
"$MERXEN_TEST_PYTHON" -c \
  'from merxen.segmentation.model_identity import require_cellpose_runtime; print(require_cellpose_runtime())'

nextflow run workflows/acceptance/hitl0_tile_check.nf \
  -ansi-log false -work-dir "$MERXEN_CHECK_ROOT/work" \
  --source_root "$MERXEN_TEST_CHECKOUT" \
  --python "$MERXEN_TEST_PYTHON" \
  --config /srv/storage/MerXen/work/d2/2bb63695bfa15cdb697b604cb6f624/segment_config.json \
  --source_zarr /srv/storage/MerXen/work/c9/800bf12cfc879ab1368ba767fff599/spatialdata_out/source_spatialdata.zarr \
  --baseline_mask /srv/storage/MerXen/results/P7513/xenium/segmentation/cellpose_masks_tiled.npy \
  --baseline_stats /srv/storage/MerXen/results/P7513/xenium/segmentation/cellpose_stitching_stats.json \
  --protected_root /srv/storage/MerXen/results \
  --gpu_lock_file /tmp/merxen-dwight-gpu.lock \
  --image_key morphology_focus --tile_index 44 \
  --outdir "$MERXEN_CHECK_ROOT/report"

export MERXEN_TILE_REPORT="$MERXEN_CHECK_ROOT/report/tile_check_out/tile_check.json"
"$MERXEN_TEST_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

report = json.loads(Path(os.environ["MERXEN_TILE_REPORT"]).read_text())
print(json.dumps(report, indent=2))
assert report["baseline_unchanged"], "Published baseline checksum changed"
assert report["passed"], "Tile agreement requires investigation"
PY
```

The workflow waits for the shared GPU lock. It checks for at least 16 GiB of free
GPU memory, uses the recorded 6144 px tile grid and 256 px overlap, and preserves
the original channels, normalization and filters. It does not run ProSeg.
The scoring window excludes 656 px from every tile-core edge and drops labels
cut by the scoring-window border. Label numbers are remapped before comparison.

Expected report fields:

| Field | Expected |
|---|---|
| `model_identity.status` | `verified` |
| `model_identity.model_name` | `cpsam_v2` |
| `model_identity.cellpose_version` | `4.2.1.1` |
| `model_identity.torch_version` | `2.14.x`, including the actual CUDA build suffix |
| `model_identity.weights_sha256` | `0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667` |
| `agreement.ap_at_05` | At least `0.99` |
| `agreement.mean_matched_iou` | At least `0.99` |
| `baseline_unchanged` / `passed` | Both `true` |

The two overlap requirements make the plan's expected 0.99 agreement explicit:
AP penalizes missing and extra cells, while mean matched IoU checks their shapes.
This is a drift diagnostic against a published stock mask, not a claim that the
published mask is biological ground truth. The report also contains counts,
foreground IoU, tile coordinates, timing and peak GPU allocation.

Nextflow publishes completed diagnostic reports even when agreement is below
the threshold; its successful exit alone does not mean scientific acceptance.
The final Python assertion above checks that acceptance explicitly. A lower
score needs investigation before training. Preserve the report and the task's
`.command.log`; avoid changing thresholds just to make the check pass.

The source path above is the surviving source used by the published run. If it
has been cleaned up, stop and validate an equivalent native image in the durable
latest zarr before substituting it. Each repeat needs a fresh output directory.

## 3. Container validation: deferred

The agreed acceptance path for this milestone is Dwight's existing Conda
environment, using steps 1 and 2 above. Container validation is deferred and is
not an acceptance gate for HITL-0 or a prerequisite for starting HITL-1.

MerXen's container execution profile uses **Apptainer with SIF images**, and the
`gpu` profile adds `--nv` to GPU processes. Dwight's local workflow uses
`-profile dwight,conda`. See [Running on a cluster](../running-the-pipeline.md#running-on-a-cluster)
and [the execution profiles](../../workflows/nextflow.config).

The repository's Dockerfile is an existing image-build recipe; it does not make
Docker a requirement for running the pipeline. The earlier instructions to run
`sudo docker` were incorrect for this deployment and have been withdrawn.
Apptainer can consume OCI registry images directly without a local Docker daemon;
see [Apptainer's OCI support](https://apptainer.org/docs/user/main/docker_and_oci.html).

When container work resumes, choose an Apptainer host and an unprivileged build
route, build or obtain a refreshed SIF, and run the checkpoint/runtime and GPU
smoke tests with `apptainer exec --nv`. Neither Apptainer nor Singularity is on
Dwight's current PATH. The existing site profile still names an older CUDA 12.6
SIF; that artifact has not been rebuilt or validated against the new pins.
Do not use it for this milestone's fresh-inference checks. The CUDA 13.0
Dockerfile change remains unvalidated until the container work resumes.

## 4. Begin testing HITL-1 when its implementation branch is ready

The following is the acceptance checklist for the upcoming implementation. Its
custom-model parameters and fixture bundles are not available in HITL-0.

First run the new model-selection, preflight, cache-identity and shared-tile tests.
They must establish all of these outcomes before a real custom-model run:

| Scenario | Required behavior |
|---|---|
| No model or override parameters | Legacy config JSON unchanged; stock `cpsam_v2` selected |
| Legacy stock outputs with unknown provenance | Reused as `unverified`; no forced regeneration |
| Valid custom model and matching metadata | Selected only for its requested platform; nuclei stay stock |
| Candidate model without explicit opt-in | Refused before inference |
| Missing weights, wrong checksum, rejected status | Refused before inference |
| Wrong channels, species, platform, resolution or runtime | Refused before inference |
| Model/configuration differs from cached mask identity | Refused; existing artifacts unchanged |
| Custom-model output targets the protected baseline, including a symlink alias | Refused; `--force-rerun` must not bypass this protection |
| Shared tile function replacing the existing loop body | Identical synthetic masks, probability logits and stitching statistics |

Then use P7513 in this order:

1. Preview the ordinary stock configuration and compare it byte for byte with
   the saved control configuration. Preview must schedule no segmentation.
2. Preview a valid custom-model fixture with explicit settings and a fresh arm
   outdir outside the protected baseline. Verify the selected model, channel
   order, pinned tile size and effective overrides.
3. Try the intentionally invalid fixture variants above. Every case must fail
   before GPU inference or output replacement; verify baseline checksums.
4. Run a single tile through the shared production tile function, then repeat
   with identical settings to check reuse. Change the model identity and confirm
   that reuse is refused.

The HITL-1 change will include the exact fixture locations and executable commands
for these steps. A stock checkpoint may be wrapped in a clearly marked test
manifest to test the plumbing; that does not establish that a fine-tuned model
is accepted. Real annotation begins with HITL-2's pilot package.
