"""MECR reference tasks of a mixed samplesheet (Atera plan AT-8, WP3).

XENIUM and MERSCOPE rows share ``MECR_REFERENCE`` over their merged panels;
ATERA rows get their own ``MECR_REFERENCE_ATERA`` (the streaming reference).
The harness runs the real MECR processes and the MECR wiring of the
``main.nf`` workflow block (cut between text anchors) with a fake ``merxen``
that keeps each task's config JSON. A mixed XENIUM + ATERA samplesheet must run
two reference tasks, hand each MECR task its own reference, and render the
XENIUM reference JSON byte for byte as a XENIUM-only samplesheet does (the
cache invariant of the existing rows).
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

_HERE = Path(__file__).parent
_WORKFLOWS = _HERE.parents[1] / "workflows"
_HARNESS = _HERE / "harness"
_WIRING_START = "    mecr_samples_by_reference_ch = mecr_samples_ch\n"
_WIRING_END = "    mecr_results_ch = MECR(mecr_inputs_ch)\n"

pytestmark = [
    pytest.mark.nextflow,
    pytest.mark.skipif(shutil.which("nextflow") is None, reason="no nextflow"),
]


def _golden_module() -> ModuleType:
    """The golden test module, for its harness helpers (no sibling import)."""
    name = "_merxen_platform_config_golden"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, _HERE / "test_platform_config_golden.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _wiring() -> str:
    """The MECR wiring of the main.nf workflow block."""
    main = (_WORKFLOWS / "main.nf").read_text()
    assert main.count(_WIRING_START) == 1 and main.count(_WIRING_END) == 1
    start = main.index(_WIRING_START)
    assert start > main.index("\nworkflow {")
    return main[start : main.index(_WIRING_END) + len(_WIRING_END)]


def _script() -> str:
    tail = (_HARNESS / "mecr_reference_split.nf").read_text()
    assert tail.count("//MECR_WIRING\n") == 1
    return "\n".join(
        [
            "include { MECR_REFERENCE; MECR_REFERENCE_ATERA; MECR } "
            'from "./modules/mecr"',
            *_golden_module().workflow_functions(),
            tail.replace("//MECR_WIRING\n", _wiring()),
        ]
    )


def _launch(root: Path, rows: str, outdir: Path) -> dict[str, list[list[str]]]:
    """Run the harness for one samplesheet; return output lines by process."""
    nextflow = shutil.which("nextflow")
    assert nextflow is not None
    work_dir = root / rows
    (work_dir / "bin").mkdir(parents=True)
    shutil.copy2(_HARNESS / "mecr_split" / "merxen", work_dir / "bin" / "merxen")
    (work_dir / "h.nf").write_text(_script())
    (work_dir / "lib").symlink_to(_WORKFLOWS / "lib", target_is_directory=True)
    (work_dir / "modules").symlink_to(_WORKFLOWS / "modules", target_is_directory=True)
    result = subprocess.run(
        [
            nextflow,
            "run",
            "h.nf",
            "-c",
            str(_WORKFLOWS / "nextflow.config"),
            "-c",
            str(_HARNESS / "mecr_split" / "resources.config"),
            "-ansi-log",
            "false",
            "--outdir",
            str(outdir),
            "--harness_rows",
            rows,
        ],
        cwd=work_dir,
        capture_output=True,
        text=True,
        env=_golden_module().hermetic_env(),
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, f"{result.stdout[-3000:]}\n{result.stderr[-2000:]}"
    outputs: dict[str, list[list[str]]] = {}
    for line in result.stdout.splitlines():
        if line.startswith("OUT\t"):
            _, process, *fields = line[: line.index("\tEND")].split("\t")
            outputs.setdefault(process, []).append(fields)
    return outputs


@pytest.fixture(scope="module")
def launches(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict]:
    root = tmp_path_factory.mktemp("mecr_split")
    # Both launches share one absolute outdir: it feeds the reference cache dir.
    outdir = root / "results"
    return {rows: _launch(root, rows, outdir) for rows in ("xenium", "mixed")}


def _reference_config(outputs: dict, process: str) -> tuple[bytes, dict]:
    ((out_dir,),) = outputs[process]
    path = Path(out_dir) / "mecr_reference_config.json"
    return path.read_bytes(), json.loads(path.read_text())


def test_mixed_samplesheet_runs_two_reference_tasks(launches: dict) -> None:
    mixed = launches["mixed"]
    assert len(mixed["MECR_REFERENCE"]) == 1
    assert len(mixed["MECR_REFERENCE_ATERA"]) == 1
    _, standard = _reference_config(mixed, "MECR_REFERENCE")
    _, atera = _reference_config(mixed, "MECR_REFERENCE_ATERA")
    assert [sample["platform"] for sample in standard["samples"]] == ["XENIUM"]
    assert [sample["platform"] for sample in atera["samples"]] == ["ATERA"]
    # The ATERA reference streams; the XENIUM one keeps the old JSON (no key).
    assert atera["reference_mode"] == "streaming"
    assert atera["reference_workers"] == 1  # task.cpus (1 in this harness)
    assert "reference_mode" not in standard
    assert "reference_workers" not in standard
    # Same atlas inputs for both references.
    for key in ("reference_atlas", "marker_min_target_fraction", "taxonomy_level"):
        assert atera[key] == standard[key], key


def test_xenium_reference_json_is_unchanged_by_an_atera_row(launches: dict) -> None:
    alone, _ = _reference_config(launches["xenium"], "MECR_REFERENCE")
    mixed, _ = _reference_config(launches["mixed"], "MECR_REFERENCE")
    assert mixed == alone
    assert "MECR_REFERENCE_ATERA" not in launches["xenium"]


def test_each_mecr_task_gets_its_own_reference(launches: dict) -> None:
    used = {
        pair_id: (Path(out_dir) / "reference_used.txt").read_text().strip()
        for out_dir, pair_id, _segmentation in launches["mixed"]["MECR"]
    }
    assert used == {"PAIR_B": "standard reference", "PAIR_A": "atera reference"}
    alone = {
        pair_id: (Path(out_dir) / "mecr_config.json").read_bytes()
        for out_dir, pair_id, _segmentation in launches["xenium"]["MECR"]
    }
    mixed = {
        pair_id: (Path(out_dir) / "mecr_config.json").read_bytes()
        for out_dir, pair_id, _segmentation in launches["mixed"]["MECR"]
    }
    assert mixed["PAIR_B"] == alone["PAIR_B"]
