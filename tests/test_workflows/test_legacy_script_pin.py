"""Pin the legacy clustering Nextflow text (plan §2.2, §13.2; R-eng E-M1).

Nextflow's ``-resume`` cache key of a task covers its script text, and the
legacy clustering results follow from that text, its wiring in ``main.nf``
and the param defaults it reads. Until both species flip to ``map_first``,
no milestone may change any of them: the sha256 digests below must stay
equal.

How to update the pin deliberately: a change here invalidates the ``-resume``
cache of every legacy clustering run and may change legacy results. If it is
intended, put the change and the new digest in their own commit whose message
states the reason and the expected effect on legacy runs; the failure message
prints the new digest (``python tests/test_workflows/test_legacy_script_pin.py``
prints all of them).
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / "workflows"
CLUSTERING_MODULE = WORKFLOWS / "modules" / "clustering_squidpy.nf"
MAIN_NF = WORKFLOWS / "main.nf"
PARAM_FILES = (WORKFLOWS / "nextflow.config", WORKFLOWS / "conf" / "dwight.config")
LEGACY_PROCESSES = (
    "CLUSTERING_SQUIDPY_PREPARE",
    "CLUSTERING_SQUIDPY_COMPUTE",
    "CLUSTERING_SQUIDPY_FINALIZE",
)

EXPECTED_SHA256: dict[str, str] = {
    "CLUSTERING_SQUIDPY_PREPARE": (
        "f415e43e13f9d711b89f0a384343a912c3d972b6663458526c67e7b2a8365d5c"
    ),
    "CLUSTERING_SQUIDPY_COMPUTE": (
        "c6fe538637d65f369deb83a4a50269241af76106d4a6ae73d0af7379f8ae2bfc"
    ),
    "CLUSTERING_SQUIDPY_FINALIZE": (
        "f432a219dc76a850d0faac2261fe02fe0e6e39b2aab4ab1a2706f22c6f84c8d3"
    ),
    # M5 (hook H5): the legacy PREPARE -> COMPUTE statements moved into the
    # else-branch of the map_first switch; process texts and defaults are
    # unchanged, so legacy task hashes, -resume and the legacy DAG are too.
    "main.nf wiring": (
        "f476ca3c9babb8a4acbadf692f760058dc8580e1701b7625212e78511ace5623"
    ),
    "param defaults": (
        "fe4ca1ed06ba1c74dba06b3a50219cfac58a719376ead6792ae7ae0568c5f3c2"
    ),
}


def process_block(module_text: str, name: str) -> str:
    """Return one process definition, from its header to its closing brace.

    Args:
        module_text: Source of a Nextflow module.
        name: Process name.

    Returns:
        The definition text (comments after its closing brace excluded).
    """
    starts = [match.start() for match in re.finditer(r"^process ", module_text, re.M)]
    header = re.search(rf"^process {name} \{{\n", module_text, re.MULTILINE)
    assert header is not None, f"process {name} not found"
    end = min((start for start in starts if start > header.start()), default=None)
    chunk = module_text[header.start() : end]
    closing = chunk.rstrip().rfind("\n}")
    assert closing >= 0, f"no closing brace for {name}"
    return chunk[: closing + 2]


def main_nf_wiring(main_text: str) -> str:
    """Return the ``main.nf`` include and calls of the legacy processes.

    Args:
        main_text: Source of ``workflows/main.nf``.

    Returns:
        The include statement of the three processes followed by the lines
        from ``clustering_inputs_ch =`` through the FINALIZE call.
    """
    include = re.search(
        r'^include \{[^}]*\} from "\./modules/clustering_squidpy"$',
        main_text,
        re.MULTILINE,
    )
    assert include is not None, "legacy clustering include not found"
    for name in LEGACY_PROCESSES:
        assert name in include.group(0), name
    start = main_text.index("    clustering_inputs_ch =\n")
    finalize = "CLUSTERING_SQUIDPY_FINALIZE(clustering_computed_ch)\n"
    end = main_text.index(finalize, start) + len(finalize)
    return f"{include.group(0)}\n{main_text[start:end]}"


def _param_assignments(config_text: str) -> dict[str, str]:
    """Return ``name -> assignment line`` of a config's ``params`` block."""
    match = re.search(r"^params \{\n(?P<body>.*?)^\}", config_text, re.M | re.S)
    assert match is not None, "params block not found"
    lines: dict[str, str] = {}
    for line in match.group("body").splitlines():
        assignment = re.match(r"^    (\w+)\s*=", line)
        if assignment:
            lines[assignment.group(1)] = line.strip()
    return lines


def legacy_param_defaults() -> str:
    """Return the base and Dwight defaults of the params the processes read.

    Returns:
        One ``<file>: <assignment>`` line (or ``<file>: <name> unset``) per
        referenced param and config file, sorted by param name.
    """
    module_text = CLUSTERING_MODULE.read_text()
    blocks = "\n".join(process_block(module_text, name) for name in LEGACY_PROCESSES)
    names = sorted(set(re.findall(r"\bparams\.(\w+)", blocks)))
    lines = []
    for name in names:
        for path in PARAM_FILES:
            assignments = _param_assignments(path.read_text())
            value = assignments.get(name, f"{name} unset")
            lines.append(f"{path.relative_to(WORKFLOWS)}: {value}")
    return "\n".join(lines)


def pinned_texts() -> dict[str, str]:
    """Return every pinned text, keyed as in ``EXPECTED_SHA256``.

    Returns:
        Pin name to text.
    """
    module_text = CLUSTERING_MODULE.read_text()
    texts = {name: process_block(module_text, name) for name in LEGACY_PROCESSES}
    texts["main.nf wiring"] = main_nf_wiring(MAIN_NF.read_text())
    texts["param defaults"] = legacy_param_defaults()
    return texts


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_legacy_clustering_text_is_pinned() -> None:
    """The legacy clustering script text, wiring and defaults are unchanged."""
    actual = {name: _sha256(text) for name, text in pinned_texts().items()}
    changed = {
        name: digest
        for name, digest in actual.items()
        if digest != EXPECTED_SHA256[name]
    }

    assert set(actual) == set(EXPECTED_SHA256)
    assert not changed, (
        f"Legacy clustering Nextflow text changed: {sorted(changed)}. This "
        "invalidates legacy -resume and may change legacy results (plan §2.2). "
        "Revert it, or, if deliberate, set EXPECTED_SHA256 in "
        f"{Path(__file__).relative_to(REPO_ROOT)} to {changed} in a separate "
        "commit that states the reason."
    )


def test_pinned_texts_cover_the_legacy_scripts() -> None:
    """Each pinned block holds its process's script and the calls."""
    texts = pinned_texts()

    for name in LEGACY_PROCESSES:
        assert texts[name].startswith(f"process {name} {{")
        assert texts[name].endswith("\n}")
        assert '"""' in texts[name]
    assert "clustering_squidpy_stages prepare" in texts["CLUSTERING_SQUIDPY_PREPARE"]
    assert "clustering_squidpy_stages compute" in texts["CLUSTERING_SQUIDPY_COMPUTE"]
    assert "clustering_squidpy_stages finalize" in texts["CLUSTERING_SQUIDPY_FINALIZE"]
    for name in LEGACY_PROCESSES:
        assert f"= {name}(" in texts["main.nf wiring"]
    assert "clustering_squidpy_min_counts = 10" in texts["param defaults"]
    assert "clustering_squidpy_use_gpu = true" in texts["param defaults"]


if __name__ == "__main__":
    for pin_name, pin_text in pinned_texts().items():
        sys.stdout.write(f"{pin_name}: {_sha256(pin_text)}\n")
