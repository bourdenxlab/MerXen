# Workstation task scheduling and streaming analysis plan

Status: proposed, 2026-11-20. Implementation is deferred by request. This document
changes no process, resource setting, scientific parameter or active run.

## Problem and evidence

Completed samples should advance through CPU processing while another sample
uses the GPU. The current workstation launch unnecessarily queues ready ProSeg
tasks behind Cellpose work.

The inspected run executes pinned release `v1.0.2`; the relevant resource and
workflow definitions are also present in `v1.1.1`. Its private launch config
allows 48 CPU slots and 460 GB RAM. These are observations of that launch, not
proposed universal defaults:

| Process | CPU slots | Requested RAM |
| --- | ---: | ---: |
| `CELLPOSE_SEGMENT` | 12 | 212 GB |
| `CELLPOSE_NUCLEI_SEGMENT` | 12 | 212 GB |
| `PROSEG_SEGMENT` | 32 | 220 GB |

The cell and nuclei processes each admit a task. Their shared GPU lock is
acquired in `beforeScript`, after Nextflow has reserved resources. One task
computes while the other waits for the lock; both retain their reservations.
Together they reserve 24 CPU slots and 424 GB, leaving 24 slots and 36 GB.
ProSeg cannot fit despite substantial unused physical memory. One Cellpose task
plus one ProSeg task would request 44 slots and 432 GB and fit that allowance.

The existing `maxForks` settings apply separately to each process; setting both
to one does not establish a shared GPU admission limit. The lock correctly
protects the GPU, including across launches, but does not tell Nextflow that a
waiting task is not doing useful work.

Actual memory use must still guide any change. For example, a completed
MERSCOPE cell task recorded about 142 GB peak RSS. A small Xenium task's lower
peak is not evidence that all MERSCOPE reservations can be reduced safely.

The workflow already joins cell and nuclei outputs by dataset before submitting
ProSeg. Ready ProSeg tasks are visible in the submission queue, so their
immediate delay is resource admission, not an all-sample segmentation barrier.

## Dependencies to preserve

- Each dataset's current ProSeg stage needs its own cell and nuclear masks,
  including inputs used to construct hybrid segmentation.
- Paired registration and comparisons need the required outputs of both
  platforms for that pair and segmentation branch, not every pair in the run.
- Stages that update the same SpatialData store must retain their current
  ordering and write protection. Parallelism across separate datasets does not
  justify overlapping writers to one store.
- Shared MECR reference preparation currently gathers the cohort's panels.
  Keep that collection explicit and local to MECR; unrelated analyses should
  not inherit its barrier.
- Cellpose also performs CPU and memory intensive work. ProSeg and image/store
  processing share CPUs, RAM and storage bandwidth, so concurrency stays
  bounded even when the GPU is available.

## Proposed implementation

### 1. Record the scheduling baseline

Capture resolved resource limits, requested resources, task start/end times,
GPU lock waits, peak memory and storage activity in private evidence under
`$MERXEN_EVIDENCE_ROOT`. Record the pinned code revision and runtime version.
Distinguish time waiting for GPU access from actual computation. Use aliases in
any committed summaries and keep input locations in local configuration.

### 2. Make GPU admission visible to scheduling

Permit at most one heavy GPU task from a launch to hold CPU/RAM reservations
while preserving capacity for one CPU ProSeg task. Cover cell and nuclei
segmentation together and audit the other GPU consumers, including training,
registration and optional GPU clustering.

Evaluate a shared Cellpose task process with an explicit cell/nuclei operation
against native GPU allocation in a separately validated newer Nextflow runtime.
The inspected run uses Nextflow 26.04.4; native local-executor accelerator
allocation was added in 26.10 and cannot simply be enabled in that older run.
Choose a design after a small admission test, not by upgrading a live runtime.

Retain the operating-system GPU lock for protection across separate launches.
A scheduler-visible GPU slot in one launch does not coordinate another launch.
Test that case explicitly, including whether an external lock owner can again
leave an expensive waiter reserving resources.

Raising the executor allowance is a possible measured operational workaround,
not the default fix. Two current Cellpose reservations plus one ProSeg request
need 56 CPU slots and 644 GB. Any larger allowance must leave physical headroom
for the operating system, Nextflow and other jobs; it must not be inferred from
CPU count alone. Do not disable the GPU lock or remove memory requests.

### 3. Release completed samples and pairs promptly

Audit grouping and collection operators on paths from segmentation to terminal
analysis. In particular, `analysis_samples_from_dataset_ch` currently uses
`groupTuple()` without an expected size. That standalone route waits for its
input channel to close, which delays ready single-platform samples.

Use the expected active-platform count for each sample/branch, with `groupKey`
where counts vary, so a group can emit when its own inputs arrive. Validate
distinct platform identities and reject duplicate or incomplete groups.
Do not emit an incomplete paired comparison as a success.

Do not replace grouping operators indiscriminately: existing completion gates
that already use `groupKey` can emit early despite calling `groupTuple()`.
Preserve those gates and the intentionally cohort-wide MECR collection.

### 4. Bound CPU and storage concurrency

Measure one Cellpose task running with one ProSeg task before increasing other
concurrency. Check both peak memory and elapsed time: sharing CPU or storage
can slow segmentation even when GPU memory fits. Tune executor limits and
per-stage forks through configuration, with conservative defaults and clear
headroom. Extend overlap to enrichment, viewer preparation and analysis only
when their own inputs and dataset write dependencies are satisfied.

### 5. Validate before rollout

Use small controlled workflow tasks to demonstrate that:

- A fast sample starts ProSeg before a different slow sample finishes Cellpose.
- At most one heavy GPU task per launch is admitted, and cross-launch GPU
  exclusion still holds.
- Task reservations remain within the configured CPU and memory allowance.
- Single-platform groups emit at one platform and paired groups at both;
  incomplete or failed branches never release a partial paired analysis.
- Independent pairs advance without waiting for the cohort, while MECR retains
  its documented cohort dependency.
- Store writers stay ordered, interrupted tasks retry cleanly and completed
  compatible tasks can be reused.

Then run one bounded representative overlap pilot. Compare scientific inputs,
model identities, segmentation settings and resulting object/assignment
identities with the sequential route. Report any numerical differences rather
than assuming concurrency is scientifically neutral. Full-section reruns are
not needed merely to test scheduler admission.

## Rollout and cache compatibility

Implement later on a short-lived branch and merge after the focused checks and
CI pass. Document any new scheduling/runtime requirement and its fallback.
Changing process names, task scripts or CPU-derived thread settings can change
task hashes. Record the expected cache impact; do not promise reuse across a
workflow redesign.

Keep the current long run on its original pinned code, path and configuration.
Do not apply this plan to it, including during a maintenance restart, without
a separate decision to accept the resulting cache impact. See
[worktrees and running pipelines](../development.md#worktrees-and-running-pipelines).

## Maintenance interruption and resume

For a reboot, stop and later resume the run; do not suspend its processes with
`Ctrl-Z` and expect them to survive power loss.

1. Identify and stop any external automatic-retry or continuation helper first.
   Confirm it exited before stopping Nextflow, so it cannot relaunch the run
   when the launcher lock is released.
2. Send one normal interrupt or termination request to the exact Nextflow
   controller for this run. Let it cancel active local tasks and close its
   cache. Verify that the controller and its task processes have exited; an
   absent controller alone does not prove that no task children remain.
3. Preserve the launch directory's `.nextflow` cache/history, the entire work
   directory, published results, pinned source checkout, configs, samplesheet,
   model files, repaired inputs and annotation backups. Published symlinks also
   depend on their work-directory targets. Do not run `nextflow clean`.
4. After a clean stop, back up the small launch/cache metadata and flush pending
   writes before an ordinary operating-system shutdown. Keep the original work
   and data locations intact; a metadata backup is not a backup of their data.
5. After reboot, verify storage mounts at the same paths and the unchanged
   pinned source revision, model and configuration. Verify annotation hashes
   against the existing backups. A reboot removes tmux sessions, so recreate
   the session instead of assuming it survived.
6. Resume from the same launch directory and pinned code path with the same
   inputs/work directory and an explicit recorded session ID. Inspect cache
   reuse and the first restarted task before leaving it unattended.

Completed tasks are reusable when their hashes, successful exit records and
required outputs remain valid. Incomplete tasks may restart from the beginning;
do not assume a partially written Cellpose mask is a completed checkpoint.
Any previously failed task with a now-repaired input is also retried by this
manual resume, so its old one-off retry helper need not be rearmed solely for
that historical failure. Review any other helper's purpose separately.

## Source references

- Repository: `workflows/nextflow.config` process reservations;
  `workflows/conf/dwight.config` GPU locks and process concurrency;
  `workflows/modules/segmentation.nf` per-dataset ProSeg inputs;
  `workflows/main.nf` standalone analysis grouping and MECR collection.
- [Nextflow local executor and accelerator support](https://docs.seqera.io/nextflow/reference/executor/local).
- [Nextflow grouping and expected group sizes](https://docs.seqera.io/nextflow/reference/operator#grouptuple).
- [Nextflow caching and resuming](https://docs.seqera.io/nextflow/cache-and-resume).
- [Nextflow signal handling](https://github.com/nextflow-io/nextflow/blob/v26.04.4/modules/nextflow/src/main/groovy/nextflow/Session.groovy).
