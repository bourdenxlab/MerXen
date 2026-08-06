process GASTON_PREPARE {
    tag "${pair_id}:${platform}:${segmentation}"

    input:
    tuple val(branch_key),
        val(pair_id),
        val(segmentation),
        val(platform),
        val(sample_id),
        val(config_json),
        path(clustered_h5ad, stageAs: "clustered_input.h5ad"),
        val(latest_zarr),
        val(n_restarts),
        val(use_gpu),
        val(_terminal_token)

    output:
    tuple val(branch_key),
        val(pair_id),
        val(segmentation),
        val(platform),
        val(sample_id),
        path("gaston_config.json"),
        path("gaston_prepare_out"),
        val(latest_zarr),
        val(n_restarts),
        val(use_gpu)

    script:
    """
    set -euo pipefail
    export PYTHONPATH="${projectDir}/../src:\${PYTHONPATH:-}"
    export OMP_NUM_THREADS="${task.cpus}"
    export OPENBLAS_NUM_THREADS="${task.cpus}"
    export MKL_NUM_THREADS="${task.cpus}"
    export NUMEXPR_NUM_THREADS="${task.cpus}"

    cat > gaston_config.json <<JSON
${config_json}
JSON
    python -m merxen.gaston_stages prepare \
        --config gaston_config.json \
        --output-dir gaston_prepare_out
    """
}

process GASTON_GLM_PCA {
    tag "${pair_id}:${platform}:${segmentation}"

    input:
    tuple val(branch_key),
        val(pair_id),
        val(segmentation),
        val(platform),
        val(sample_id),
        path(gaston_config),
        path(prepared_dir),
        val(latest_zarr),
        val(n_restarts),
        val(use_gpu)

    output:
    tuple val(branch_key),
        val(pair_id),
        val(segmentation),
        val(platform),
        val(sample_id),
        path(gaston_config),
        path(prepared_dir),
        path("gaston_glmpca_out"),
        val(latest_zarr),
        val(n_restarts),
        val(use_gpu)

    script:
    """
    set -euo pipefail
    export PYTHONPATH="${projectDir}/../src:\${PYTHONPATH:-}"
    export CUDA_VISIBLE_DEVICES=""
    export OMP_NUM_THREADS="${task.cpus}"
    export OPENBLAS_NUM_THREADS="${task.cpus}"
    export MKL_NUM_THREADS="${task.cpus}"
    export NUMEXPR_NUM_THREADS="${task.cpus}"

    python -m merxen.gaston_stages glmpca \
        --config "${gaston_config}" \
        --bundle-dir "${prepared_dir}" \
        --output-dir gaston_glmpca_out
    """
}

process GASTON_TRAIN {
    tag "${pair_id}:${platform}:${segmentation}:seed${seed}"

    input:
    tuple val(group_key),
        val(pair_id),
        val(segmentation),
        val(platform),
        val(sample_id),
        path(gaston_config),
        path(prepared_dir),
        path(glmpca_dir),
        val(latest_zarr),
        val(seed),
        val(use_gpu)

    output:
    tuple val(group_key), path("seed_${seed}")

    script:
    """
    set -euo pipefail
    export PYTHONPATH="${projectDir}/../src:\${PYTHONPATH:-}"
    export OMP_NUM_THREADS="${task.cpus}"
    export OPENBLAS_NUM_THREADS="${task.cpus}"
    export MKL_NUM_THREADS="${task.cpus}"
    export NUMEXPR_NUM_THREADS="${task.cpus}"

    # Keep binary-runtime failures outside the rankable per-seed exception
    # handler. A broken shared Torch environment must fail the Nextflow task
    # instead of producing a cacheable failed-seed manifest for every restart.
    python -c "import torch; from gaston import neural_net; requested = '${use_gpu}' == 'true'; assert not requested or torch.cuda.is_available(), 'GASTON training requested CUDA but torch.cuda is unavailable'; print('GASTON training runtime:', torch.__version__, 'CUDA available:', torch.cuda.is_available())"

    if ${params.gaston_gpu_vram_monitor} && [[ "${use_gpu}" == "true" ]]; then
        mkdir -p "seed_${seed}/gpu_vram"
        python -m merxen.gaston_stages train \
            --config "${gaston_config}" \
            --bundle-dir "${prepared_dir}" \
            --glmpca-dir "${glmpca_dir}" \
            --seed "${seed}" \
            --output-dir "seed_${seed}" &
        gaston_pid=\$!
        python -m merxen.monitoring.gpu_vram \
            --pid "\${gaston_pid}" \
            --interval-seconds ${params.gaston_gpu_vram_monitor_interval_seconds} \
            --samples-path "seed_${seed}/gpu_vram/samples.tsv" \
            --summary-path "seed_${seed}/gpu_vram/summary.json" &
        monitor_pid=\$!
        set +e
        wait "\${gaston_pid}"
        gaston_exit=\$?
        wait "\${monitor_pid}" || true
        set -e
        exit "\${gaston_exit}"
    fi

    python -m merxen.gaston_stages train \
        --config "${gaston_config}" \
        --bundle-dir "${prepared_dir}" \
        --glmpca-dir "${glmpca_dir}" \
        --seed "${seed}" \
        --output-dir "seed_${seed}"
    """
}

process GASTON_POSTPROCESS {
    tag "${pair_id}:${platform}:${segmentation}"

    publishDir { "${params.outdir}/${pair_id}/${segmentation}/gaston" }, mode: "copy", overwrite: true

    input:
    tuple val(branch_key),
        val(pair_id),
        val(segmentation),
        val(platform),
        val(sample_id),
        val(gaston_config_json),
        path(prepared_dir),
        path(glmpca_dir),
        val(latest_zarr),
        path(seed_dirs)

    output:
    tuple val(branch_key),
        val(pair_id),
        val(segmentation),
        val(platform),
        val(sample_id),
        path("gaston_postprocess_config.json"),
        path("gaston_out"),
        val(latest_zarr)

    script:
    def seedArguments = seed_dirs.collect { seedDir -> "--seed-dir '${seedDir}'" }.join(" ")
    """
    set -euo pipefail
    export PYTHONPATH="${projectDir}/../src:\${PYTHONPATH:-}"
    export CUDA_VISIBLE_DEVICES=""
    export OMP_NUM_THREADS="${task.cpus}"
    export OPENBLAS_NUM_THREADS="${task.cpus}"
    export MKL_NUM_THREADS="${task.cpus}"
    export NUMEXPR_NUM_THREADS="${task.cpus}"

    cat > gaston_postprocess_config.json <<JSON
${gaston_config_json}
JSON

    python -c "import torch; from gaston import dp_related, model_selection; print('GASTON postprocessing runtime:', torch.__version__)"

    python -m merxen.gaston_stages postprocess \
        --config gaston_postprocess_config.json \
        --bundle-dir "${prepared_dir}" \
        --glmpca-dir "${glmpca_dir}" \
        ${seedArguments} \
        --output-dir "gaston_out/${platform.toLowerCase()}"
    """
}

process GASTON_IMPORT {
    tag "${pair_id}:${platform}:${segmentation}"

    publishDir { "${params.outdir}/${pair_id}/${segmentation}/gaston" }, mode: "copy", overwrite: true

    input:
    tuple val(branch_key),
        val(pair_id),
        val(segmentation),
        val(platform),
        val(sample_id),
        path(gaston_config),
        path(standalone_dir, stageAs: "gaston_standalone"),
        path(clustered_h5ad, stageAs: "clustered_input.h5ad"),
        val(latest_zarr)

    output:
    tuple val(pair_id),
        val(segmentation),
        val(platform),
        path("gaston_out")

    script:
    """
    set -euo pipefail
    export PYTHONPATH="${projectDir}/../src:\${PYTHONPATH:-}"
    export OMP_NUM_THREADS="${task.cpus}"
    export OPENBLAS_NUM_THREADS="${task.cpus}"
    export MKL_NUM_THREADS="${task.cpus}"
    export NUMEXPR_NUM_THREADS="${task.cpus}"

    python -m merxen.gaston_stages import \
        --config "${gaston_config}" \
        --standalone-dir "gaston_standalone/${platform.toLowerCase()}" \
        --output-dir "gaston_out/${platform.toLowerCase()}"
    """
}
