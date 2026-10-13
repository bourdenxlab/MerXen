#!/usr/bin/env bash
# Refuse a Nextflow launch whose resolved configuration reads from SSD1.
#
# /media/mathieubo/SSD1 is failing (annotation_dev/SSD1_DAMAGE_20261013.md and
# its addendum): most of its reference bundles and sources no longer match
# their recorded hashes, and the store reuse check compares only the size of
# files over 64 MB. No dev launch may read from it. This guard resolves the
# configuration the way the launch will, applies the launch's own --param
# overrides, and exits non-zero if any resolved value points there, before
# Nextflow starts a single task.
#
# Usage: give it the arguments of the `nextflow run` call, in the same spelling
# (a script keeps them in one array and passes it to both):
#
#   launch_guard.sh .../workflows/main.nf -profile dwight,conda -c launch.config \
#       --species mouse --annotation_reference_store /srv/... [more --params]
#
# Understood: the project (a main.nf path or its directory), -profile, -c /
# -config and every --name value / --name=value / --name (true). Other
# single-dash options (-w, -resume, ...) are ignored. -params-file is refused,
# because the guard cannot see what it sets: pass the params on the command line.
#
# What is checked: every value `nextflow config -o flat` prints, after the
# --param overrides replace the params they name. Not checked: values a config
# derives from a param inside its own expressions (the printed value is the
# default, so an override of the source param does not clear a derived value);
# a path that resolves to SSD1 through a symlink; anything a process reads
# that is not in the configuration (a samplesheet column, a bundle_ref.json).
#
# What is let through, each reported in the output:
#   * the allow-list: exact values in LAUNCH_GUARD_ALLOW (comma-separated) and
#     the params named in LAUNCH_GUARD_ALLOW_KEYS (comma-separated). The
#     default allow-list is the human gene table alone (the WHB Nonneurons
#     h5ad: only its var index is read, to test gene symbols against the human
#     gene set; its sha256 is recorded in the SSD1 addendum), and only under
#     the params that read it as a gene table (annotation_human_gene_table and
#     annotation_gene_id_fallback_csv): the same file under another param, such
#     as the legacy mecr_whb_nonneurons_h5ad_path, is a full matrix read and is
#     refused. Drop it from the list when the human table moves (plan U6).
#   * for --species mouse, the params of the human WHB and SEA-AD references
#     (names with a "whb" or "seaad" part): a mouse launch never reads them.
#
# Environment: NEXTFLOW (the executable; default nextflow), LAUNCH_GUARD_VERBOSE
# (list the skipped human-source values), LAUNCH_GUARD_PREFIX (default /media/mathieubo/SSD1), LAUNCH_GUARD_PROJECT
# (project directory when no main.nf is given; default: the repository's
# workflows/ directory next to this script).
#
# Exit status: 0 allowed, 1 refused, 2 usage error or `nextflow config` failed
# (a launch whose configuration cannot be read is not allowed either).

set -euo pipefail

PREFIX="${LAUNCH_GUARD_PREFIX:-/media/mathieubo/SSD1}"
HUMAN_GENE_TABLE="/media/mathieubo/SSD1/MerXen/mapmycells/abc_whb/expression_matrices/WHB-10Xv3/20240330/WHB-10Xv3-Nonneurons-raw.h5ad"
NEXTFLOW_BIN="${NEXTFLOW:-nextflow}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="${LAUNCH_GUARD_PROJECT:-${HERE}/../../workflows}"

die() {
    echo "launch_guard: $*" >&2
    exit 2
}

declare -A override=()
declare -a override_order=()
declare -a config_files=()
profile=""

while (($#)); do
    case "$1" in
        -profile)
            (($# > 1)) || die "-profile needs a value"
            profile="$2"
            shift 2
            ;;
        -c | -config)
            (($# > 1)) || die "$1 needs a value"
            config_files+=("$2")
            shift 2
            ;;
        -params-file | -params-file=*)
            die "-params-file is not understood (the guard cannot see its values); pass the params as --name value"
            ;;
        --*=*)
            name="${1#--}"
            key="${name%%=*}"
            [[ -v "override[$key]" ]] || override_order+=("$key")
            override["$key"]="${name#*=}"
            shift
            ;;
        --*)
            key="${1#--}"
            [[ -v "override[$key]" ]] || override_order+=("$key")
            if (($# > 1)) && [[ "$2" != -* ]]; then
                override["$key"]="$2"
                shift 2
            else
                override["$key"]="true"
                shift
            fi
            ;;
        -*)
            shift
            ;;
        *)
            if [[ "$1" == *main.nf && -f "$1" ]]; then
                project="$(dirname "$1")"
            elif [[ -f "$1/main.nf" ]]; then
                project="$1"
            fi
            shift
            ;;
    esac
done

[[ -f "${project}/main.nf" ]] || die "no main.nf in the project directory ${project}"

nextflow_args=()
for config_file in ${config_files[@]+"${config_files[@]}"}; do
    nextflow_args+=(-c "${config_file}")
done
nextflow_args+=(config)
[[ -n "${profile}" ]] && nextflow_args+=(-profile "${profile}")
nextflow_args+=(-o flat "${project}")

flat_config="$(NXF_ANSI_LOG=false "${NEXTFLOW_BIN}" "${nextflow_args[@]}")" ||
    die "'nextflow config' failed; the launch is not allowed (${NEXTFLOW_BIN} ${nextflow_args[*]})"
[[ -n "${flat_config}" ]] || die "'nextflow config' printed nothing; the launch is not allowed"

# Resolved values as "key<TAB>value": every flat line, then the overrides of
# params the config does not define. A flat value is a single-quoted string
# or a bare literal; only the quotes are stripped.
declare -A seen=()
resolved_keys=()
resolved_values=()
config_species=""
while IFS= read -r line; do
    [[ "${line}" =~ ^([^[:space:]=]+)[[:space:]]=[[:space:]](.*)$ ]] || continue
    key="${BASH_REMATCH[1]}"
    value="${BASH_REMATCH[2]}"
    if [[ "${value}" == \'*\' ]]; then
        value="${value:1:${#value}-2}"
    fi
    if [[ "${key}" == params.* ]]; then
        name="${key#params.}"
        seen["${name}"]=1
        [[ "${name}" == species ]] && config_species="${value}"
        if [[ -v "override[${name}]" ]]; then
            value="${override[${name}]}"
        fi
    fi
    resolved_keys+=("${key}")
    resolved_values+=("${value}")
done <<<"${flat_config}"
for name in ${override_order[@]+"${override_order[@]}"}; do
    [[ -v "seen[${name}]" ]] && continue
    resolved_keys+=("params.${name}")
    resolved_values+=("${override[${name}]}")
done

species="${override[species]:-${config_species}}"

GENE_TABLE_KEYS=(annotation_human_gene_table annotation_gene_id_fallback_csv)
IFS=',' read -r -a allowed_values <<<"${LAUNCH_GUARD_ALLOW:-}"
IFS=',' read -r -a allowed_keys <<<"${LAUNCH_GUARD_ALLOW_KEYS:-}"

is_listed() { # is_listed NEEDLE LIST...
    local needle="$1" item
    shift
    for item in "$@"; do
        [[ -n "${item}" && "${item}" == "${needle}" ]] && return 0
    done
    return 1
}

refused=()
allowed=()
skipped=()
for index in "${!resolved_keys[@]}"; do
    key="${resolved_keys[index]}"
    value="${resolved_values[index]}"
    [[ "${value}" == *"${PREFIX}"* ]] || continue
    name="${key#params.}"
    if { [[ "${value}" == "${HUMAN_GENE_TABLE}" && "${key}" == params.* ]] &&
        is_listed "${name}" "${GENE_TABLE_KEYS[@]}"; } ||
        is_listed "${value}" ${allowed_values[@]+"${allowed_values[@]}"} ||
        { [[ "${key}" == params.* ]] && is_listed "${name}" ${allowed_keys[@]+"${allowed_keys[@]}"}; }; then
        allowed+=("${key} = ${value}")
    elif [[ "${species}" == mouse && "${key}" == params.* && "${name}" =~ (^|_)(whb|seaad)(_|$) ]]; then
        skipped+=("${key} = ${value}")
    else
        refused+=("${key} = ${value}")
    fi
done

for entry in ${allowed[@]+"${allowed[@]}"}; do
    echo "launch_guard: allowed (allow-list): ${entry}" >&2
done
if ((${#skipped[@]})); then
    echo "launch_guard: skipped ${#skipped[@]} human WHB / SEA-AD source value(s) on ${PREFIX}: a mouse launch does not read them (LAUNCH_GUARD_VERBOSE=1 lists them)" >&2
    if [[ -n "${LAUNCH_GUARD_VERBOSE:-}" ]]; then
        for entry in "${skipped[@]}"; do
            echo "launch_guard:   ${entry}" >&2
        done
    fi
fi

if ((${#refused[@]})); then
    echo "launch_guard: REFUSED: ${#refused[@]} resolved value(s) point at ${PREFIX}, which is failing." >&2
    for entry in "${refused[@]}"; do
        echo "launch_guard:   ${entry}" >&2
    done
    echo "launch_guard: override each with --<param> <path on /srv/storage> in the launch, or, if it is not read by this launch, name it in LAUNCH_GUARD_ALLOW_KEYS." >&2
    exit 1
fi

echo "launch_guard: ok (species '${species:-unset}', ${#resolved_keys[@]} resolved values, none on ${PREFIX} outside the allow-list)" >&2
