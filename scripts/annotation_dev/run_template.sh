#!/bin/bash
# Template of a mouse map_first launch on the dwight workstation (add-only into
# published results). Copy it into the run directory, edit the CONFIGURE block,
# and keep the guard call: it refuses the launch if any resolved config value
# points at the refused disk (SSD1, which is failing; see launch_guard.sh).
#
# The shape follows the MS2 run of 2026-10-13: one array of Nextflow arguments
# is given to the guard and to `nextflow run`, so the guard checks exactly what
# the launch will use.
#
# Locations come from the private .env (template .env.example), which a code
# export does not contain: export MERXEN_ENV_FILE=<main checkout>/.env first
# (docs/local-configuration.md). The .env is read once, below; a variable the
# operator exported before the launch wins over the file (as in Python), so
# e.g. MERXEN_DWIGHT_SITE_CONFIG=none ./run.sh runs without the Dwight site
# config. Any further override belongs in the CONFIGURE block, after that line.
set -euo pipefail

: "${MERXEN_ENV_FILE:?export MERXEN_ENV_FILE=<main checkout>/.env (a code export has no private files)}"
# Plain assignments only (no wholesale export); what the operator already
# exported wins over the file, and an empty value counts as unset.
declare -A env_wins=()
for name in $(compgen -v MERXEN_); do env_wins["$name"]="${!name}"; done
. "${MERXEN_ENV_FILE}"
for name in "${!env_wins[@]}"; do
    [[ -n "${env_wins[$name]}" ]] && printf -v "$name" '%s' "${env_wins[$name]}"
done

# --- CONFIGURE --------------------------------------------------------------
R="${MERXEN_ANNOTATION_DEV_DIR}/CHANGE_ME_run_dir"            # run directory
CODE=$R/code                                                  # code export to run
OUT=CHANGE_ME_published_results_root                          # e.g. a $MERXEN_... results root
SAMPLESHEET=$R/samplesheet.csv
GUARD=${LAUNCH_GUARD:-${MERXEN_ANNOTATION_DEV_DIR}/templates/launch_guard.sh}
# ----------------------------------------------------------------------------

[[ "$R $OUT" != *CHANGE_ME* ]] || { echo "run_template: edit the CONFIGURE block first (CHANGE_ME left in it)" >&2; exit 2; }

# The private files reach the launch (and the guard, which resolves the config
# like the launch does) through these variables; a code export has none of them.
export MERXEN_ENV_FILE MERXEN_ALIAS_FILE MERXEN_SITE_CONFIG MERXEN_DWIGHT_SITE_CONFIG
export NXF_ANSI_LOG=false TMPDIR=${TMPDIR:-${MERXEN_TASK_TMP_DIR}}
mkdir -p "$R/logs" "$TMPDIR"
note() { echo "[$(date '+%F %T')] $*" | tee -a "$R/STATUS.txt"; }

# A variable that names a missing file would only surface as an unreadable
# "Unable to parse config file" (Nextflow) or a late task error: say which.
for name in MERXEN_ALIAS_FILE MERXEN_SITE_CONFIG MERXEN_DWIGHT_SITE_CONFIG; do
    value="${!name:-}"
    [[ -z "$value" || "$value" == none || "$value" == /dev/null || -f "$value" ]] || {
        note "refusing to launch: $name names a file that does not exist: $value"
        exit 1
    }
done

# Host etiquette: stop below 500 GB free on the storage RAID (an earlier run
# crashed on a full disk).
free_gb=$(df --output=avail -BG "${MERXEN_STORAGE_ROOT}" | tail -1 | tr -dc '0-9')
if ((free_gb < 500)); then
    note "refusing to launch: only ${free_gb} GB free on the storage RAID"
    exit 1
fi

# The reference stores (annotation_reference_store and its _large sibling) and
# the mouse source paths (WMB matrices and metadata, MERFISH CCF, the self-map
# test cells and the mouse gene table) come from the private Dwight site config
# (dwight.site.config), so they are not repeated here; override one only on
# purpose (build_hash covers the resolved source paths). A checkout of $CODE
# from before the store default moved still defaults the store to the refused
# disk, and the guard then refuses the launch.
#
# The legacy MapMyCells caches below are not read while every row has
# mecr_enabled false; they are pointed at an empty directory so that a row that
# enables the stage fetches fresh references instead of reading the refused disk.
args=(
    "$CODE/workflows/main.nf"
    -profile dwight,conda
    -c "$R/launch.config"
    --samplesheet "$SAMPLESHEET"
    --outdir "$OUT"
    --species mouse
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
