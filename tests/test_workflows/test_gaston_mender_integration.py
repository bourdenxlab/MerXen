"""Exercise the merged terminal selectors and the real shared release barrier."""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = [
    pytest.mark.nextflow,
    pytest.mark.skipif(
        shutil.which("nextflow") is None, reason="Nextflow is not installed"
    ),
]


def test_terminal_settings_and_all_branch_consumers_receive_their_release(
    tmp_path: Path,
) -> None:
    main = (ROOT / "workflows/main.nf").read_text()
    helpers = main[main.index("def parseChannels") : main.index("\nworkflow {")]
    defaults = {}
    for name, raw in re.findall(
        r"^    (\w+) = (.+)$", (ROOT / "workflows/nextflow.config").read_text(), re.M
    ):
        with contextlib.suppress(json.JSONDecodeError):
            defaults[name] = json.loads(raw)
    (tmp_path / "defaults.json").write_text(json.dumps(defaults))
    (tmp_path / "lib").symlink_to(ROOT / "workflows/lib", target_is_directory=True)
    barrier = main[
        main.index("    pair_terminal_specs_ch =") : main.index(
            "    mender_current_clustering_gate_ch ="
        )
    ]
    gaston_consumer = main[
        main.index("    gaston_inputs_ch =") : main.index(
            "    gaston_import_lookup_ch ="
        )
    ]
    mender_consumer = main[
        main.index("    mender_inputs_ch =") : main.index(
            "    mender_finalize_lookup_ch ="
        )
    ]
    workflow = """
workflow {
    def options = new groovy.json.JsonSlurper().parse(new File('defaults.json'))
    def cases = [
        plain: [:],
        mender: [mender_enabled: true],
        gaston: [gaston_enabled: true],
        both: [mender_enabled: true, gaston_enabled: true],
        exact: [mender_enabled: true, gaston_enabled: true, only_stage: 'gaston'],
        map: [mender_enabled: true, gaston_enabled: true, stop_stage: 'mapmycells'],
    ]
    def caseSettings = cases.collectEntries { name, controls ->
        def row = [pair_id: name, analysis_mode: 'xenium',
            analysis_segmentation: 'reseg', mender_segmentations: 'original_seg',
            gaston_segmentations: 'proseg_hybrid'] + controls
        [(name): rowSampleSettings(row, options)]
    }
    new File('settings.json').text = groovy.json.JsonOutput.toJson(caseSettings)
    // Three clustering events for one pair; two MapMyCells events plus the
    // extra clustering prerequisite for another. Published-only gets a token
    // immediately. Both terminal consumers need every acquisition branch.
    def ready = caseSettings.both + [pair_id: 'READY']
    def mapped = rowSampleSettings([pair_id: 'MAPPED', analysis_mode: 'xenium',
        analysis_segmentation: 'reseg,original_seg', mender_enabled: true,
        gaston_enabled: true, stop_stage: 'gaston'], options)
    def published = caseSettings.exact + [pair_id: 'PUBLISHED']
    sample_rows_ch = Channel.of(tuple('READY', [:], ready),
        tuple('MAPPED', [:], mapped), tuple('PUBLISHED', [:], published))
    mapmycells_results_ch = Channel.of(tuple('MAPPED', 'reseg', 'done'),
        tuple('MAPPED', 'original_seg', 'done'))
    distance_from_object_annotation_results_ch = Channel.empty()
    compute_cortical_depth_results_ch = Channel.empty()
    clustering_results_ch = Channel.fromList(['READY', 'MAPPED'].collectMany { pair ->
        ['reseg', 'original_seg', 'proseg_hybrid'].collect { seg ->
            tuple(pair, seg, '[]', 'clustered')
        }
    })
"""
    artifacts = """
    gaston_artifacts_ch = Channel.fromList(
        ['READY', 'MAPPED', 'PUBLISHED'].collectMany { pair ->
        ['reseg', 'proseg_hybrid'].collect { seg ->
            tuple(pair, "${pair}|${seg}", seg, 'XENIUM', "${pair}_XENIUM",
                file('clustered.h5ad'), '/path/to/latest', ready)
        }
    })
    mender_artifacts_ch = Channel.fromList(
        ['READY', 'MAPPED', 'PUBLISHED'].collectMany { pair ->
        ['original_seg', 'proseg_hybrid'].collect { seg ->
            tuple(pair, "${pair}|${seg}", seg, 'XENIUM', "${pair}_XENIUM",
                file('clustered.h5ad'), '/path/to/latest', ready)
        }
    })
"""
    assertions = """
    gaston_inputs_ch.map { row -> row[0].toString() }.collect().view { values ->
        new File('gaston.json').text = groovy.json.JsonOutput.toJson(values.sort())
        "gaston branches: ${values.size()}"
    }
    mender_inputs_ch.map { row -> row[0].toString() }.collect().view { values ->
        new File('mender.json').text = groovy.json.JsonOutput.toJson(values.sort())
        "mender branches: ${values.size()}"
    }
}
"""
    (tmp_path / "check.nf").write_text(
        "nextflow.enable.dsl = 2\n"
        + helpers
        + workflow
        + barrier
        + artifacts
        + gaston_consumer
        + mender_consumer
        + assertions
    )
    result = subprocess.run(
        ["nextflow", "run", "check.nf", "-ansi-log", "false"],
        cwd=tmp_path,
        env={**os.environ, "NXF_OFFLINE": "true"},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    settings = json.loads((tmp_path / "settings.json").read_text())
    for name, expected_stop in (
        ("plain", "clustering_squidpy"),
        ("mender", "mender"),
        ("gaston", "gaston"),
        ("both", "gaston"),
    ):
        assert settings[name]["stop_stage"] == expected_stop
        assert settings[name]["run_mapmycells"] is False
    assert settings["both"]["run_mender"] and settings["both"]["run_gaston"]
    assert settings["both"]["required_clustering_segmentations"] == [
        "reseg",
        "original_seg",
        "proseg_hybrid",
    ]
    assert settings["exact"]["run_gaston"] and not settings["exact"]["run_mender"]
    assert settings["exact"]["gaston_published_output_mode"]
    assert settings["map"]["run_mapmycells"]
    assert not settings["map"]["run_gaston"] and not settings["map"]["run_mender"]
    for name, segmentations in (
        ("gaston", ["reseg", "proseg_hybrid"]),
        ("mender", ["original_seg", "proseg_hybrid"]),
    ):
        branches = json.loads((tmp_path / f"{name}.json").read_text())
        assert branches == sorted(
            f"{pair}|{seg}"
            for pair in ["READY", "MAPPED", "PUBLISHED"]
            for seg in segmentations
        )
