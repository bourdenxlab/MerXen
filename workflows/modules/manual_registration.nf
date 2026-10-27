// Processes of the standalone manual landmark registration workflows
// (workflows/manual_registration_{prepare,register}.nf; docs/stages/manual-registration.md).
// cache false: the Python commands keep their own provenance and skip unchanged work in seconds,
// and the viewer edits the prepared store in place, which Nextflow's input hashing cannot see.
// Nextflow publishes nothing: Python writes the durable outputs to the absolute paths in its config.

process MANUAL_REGISTRATION_PREPARE {
    tag "${id}"
    cache false

    input:
    tuple val(id), val(prepare_config_json)

    output:
    tuple val(id), path("prepare_out")

    script:
    // CZI rows are read by the isolated CZI runtime (MerXen's CZI bridge); the sheet preflight
    // guarantees the value is set whenever a row is a .czi file.
    def czi_python = ManualRegistrationSheet.cziPython(params)
    def czi_export = (czi_python && czi_python != "in_process") ?
        "export MERXEN_CZI_PYTHON='${czi_python}'" : "# CZI read in the task environment, or no CZI rows"
    """
    set -euo pipefail
    export OMP_NUM_THREADS="${task.cpus}"
    export OPENBLAS_NUM_THREADS="${task.cpus}"
    export MKL_NUM_THREADS="${task.cpus}"
    export NUMEXPR_NUM_THREADS="${task.cpus}"
    export NUMBA_NUM_THREADS="${task.cpus}"
    export VECLIB_MAXIMUM_THREADS="${task.cpus}"
    export BLIS_NUM_THREADS="${task.cpus}"
    export DASK_NUM_WORKERS="${task.cpus}"
    ${czi_export}
    mkdir -p prepare_out

    cat > prepare_config.json <<'JSON'
${prepare_config_json}
JSON

    merxen manual-registration-prepare --config prepare_config.json
    """

    stub:
    // The stubs write the config too, so a stub run shows what a real task would be given.
    """
    mkdir -p prepare_out
    cat > prepare_config.json <<'JSON'
${prepare_config_json}
JSON
    echo '{"status": "stub"}' > prepare_out/prepare_summary.json
    """
}

process MANUAL_REGISTRATION_REGISTER {
    tag "${run_id}"
    cache false

    input:
    tuple val(run_id), val(id), val(register_config_json)

    output:
    tuple val(run_id), val(id), path("register_out")

    script:
    """
    set -euo pipefail
    export OMP_NUM_THREADS="${task.cpus}"
    export OPENBLAS_NUM_THREADS="${task.cpus}"
    export MKL_NUM_THREADS="${task.cpus}"
    export NUMEXPR_NUM_THREADS="${task.cpus}"
    export NUMBA_NUM_THREADS="${task.cpus}"
    export VECLIB_MAXIMUM_THREADS="${task.cpus}"
    export BLIS_NUM_THREADS="${task.cpus}"
    export DASK_NUM_WORKERS="${task.cpus}"
    mkdir -p register_out

    cat > register_config.json <<'JSON'
${register_config_json}
JSON

    merxen manual-registration-register --config register_config.json
    """

    stub:
    """
    mkdir -p register_out
    cat > register_config.json <<'JSON'
${register_config_json}
JSON
    echo '{"status": "stub"}' > register_out/stub.json
    """
}

process MANUAL_REGISTRATION_COMPARE {
    tag "${id}"
    cache false

    input:
    tuple val(id), val(compare_config_json)

    output:
    tuple val(id), path("compare_out")

    script:
    """
    set -euo pipefail
    export OMP_NUM_THREADS="${task.cpus}"
    export OPENBLAS_NUM_THREADS="${task.cpus}"
    export MKL_NUM_THREADS="${task.cpus}"
    export NUMEXPR_NUM_THREADS="${task.cpus}"
    export NUMBA_NUM_THREADS="${task.cpus}"
    export VECLIB_MAXIMUM_THREADS="${task.cpus}"
    export BLIS_NUM_THREADS="${task.cpus}"
    export DASK_NUM_WORKERS="${task.cpus}"
    mkdir -p compare_out

    cat > compare_config.json <<'JSON'
${compare_config_json}
JSON

    merxen manual-registration-compare --config compare_config.json
    """

    stub:
    """
    mkdir -p compare_out
    cat > compare_config.json <<'JSON'
${compare_config_json}
JSON
    """
}
