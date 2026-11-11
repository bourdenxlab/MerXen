// Optional training barrier. Stock runs never schedule this process.
process CELLPOSE_HITL_TRAIN {
    tag "named-model"
    label "gpu"
    cpus 4
    memory "32 GB"

    input:
    tuple val(training_config_json), path(annotation_packages)

    output:
    path("training_out/selector.json"), emit: model_selector

    script:
    """
    set -euo pipefail
    export OMP_NUM_THREADS="${task.cpus}"
    export MKL_NUM_THREADS="${task.cpus}"

    cat > training_config.json <<'JSON'
${training_config_json}
JSON

    merxen cellpose-hitl-train --config training_config.json --output training_out
    """
}

process CELLPOSE_HITL_SEED {
    tag "independent-inputs"
    cpus 2
    memory "4 GB"
    cache false

    input:
    tuple val(seed_config_json), path(source_store), path(nuclei_inputs)

    output:
    path("seed_complete.txt"), emit: complete

    script:
    """
    set -euo pipefail
    cat > seed_config.json <<'JSON'
${seed_config_json}
JSON
    merxen cellpose-hitl-seed --config seed_config.json
    touch seed_complete.txt
    """
}
