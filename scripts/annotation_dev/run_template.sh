#!/bin/bash
# Template of a mouse map_first launch on the dwight workstation (add-only into
# published results). Copy it into the run directory, edit the CONFIGURE block,
# and keep the guard call: it refuses the launch if any resolved config value
# points at /media/mathieubo/SSD1, which is failing (see launch_guard.sh).
#
# The shape follows the VZG2 run of 2026-10-13: one array of Nextflow arguments
# is given to the guard and to `nextflow run`, so the guard checks exactly what
# the launch will use.
set -euo pipefail

# --- CONFIGURE --------------------------------------------------------------
R=/srv/storage/MerXen/annotation_dev/CHANGE_ME_run_dir        # run directory
CODE=$R/code                                                  # checkout of the code to run
OUT=/srv/storage/mouse_vizgen_reference/merxen_results        # published results root
SAMPLESHEET=$R/samplesheet.csv
STORE=/srv/storage/MerXen/annotation_references               # reference store (SSD1 is the default)
GUARD=${LAUNCH_GUARD:-/srv/storage/MerXen/annotation_dev/templates/launch_guard.sh}
# ----------------------------------------------------------------------------

export NXF_ANSI_LOG=false TMPDIR=${TMPDIR:-/srv/storage/MerXen/tmp}
mkdir -p "$R/logs" "$STORE" "$TMPDIR"
note() { echo "[$(date '+%F %T')] $*" | tee -a "$R/STATUS.txt"; }

# Host etiquette: stop below 500 GB free on /srv/storage (an earlier run
# crashed on a full disk).
free_gb=$(df --output=avail -BG /srv/storage | tail -1 | tr -dc '0-9')
if ((free_gb < 500)); then
    note "refusing to launch: only ${free_gb} GB free on /srv/storage"
    exit 1
fi

# The mouse source paths (WMB matrices and metadata, MERFISH CCF, the self-map
# test cells and the mouse gene table) default to /srv/storage in
# dwight.annotation.config, so they are not repeated here; override one only on
# purpose, because build_hash covers the resolved source paths. The legacy
# MapMyCells caches below are not read while every row has mecr_enabled false;
# they are pointed at an empty directory so that a row that enables the stage
# fetches fresh references instead of reading SSD1.
args=(
    "$CODE/workflows/main.nf"
    -profile dwight,conda
    -c "$R/launch.config"
    --samplesheet "$SAMPLESHEET"
    --outdir "$OUT"
    --species mouse
    --annotation_reference_store "$STORE"
    --annotation_reference_store_large /srv/storage/MerXen/annotation_references_large
    --mecr_reference_cache_dir "$R/runtime_cache/mapmycells"
    --mapmycells_region_cache_dir "$R/runtime_cache/mapmycells"
    --clustering_squidpy_broad_reference_cache_dir "$R/runtime_cache/mapmycells"
)

cd "$R"
"$GUARD" "${args[@]}" || {
    note "launch guard refused the launch (exit $?)"
    exit 1
}

note "nextflow start (-resume); load $(cut -d' ' -f1-3 /proc/loadavg)"
set +e
ionice -c 2 -n 7 nice -n 10 nextflow -log "$R/logs/nextflow.log" run "${args[@]}" -w "$R/work" -resume \
    >"$R/logs/run.out" 2>&1
rc=$?
set -e
note "NEXTFLOW EXIT $rc"
exit "$rc"
