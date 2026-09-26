"""Hook points of the robust cell-type annotation effort (plan §2.4, OD-D6).

The files that ``feature/gaston`` also edits (``workflows/main.nf``, the two
workflow configs, ``src/merxen/config.py`` and ``src/merxen/io/samplesheet.py``)
change only at hook points. Each hook carries one marker ``rca-hook:H<n>``
(``//`` in Nextflow, ``#`` in Python) that occurs exactly once per file, so a
rebase or merge cannot silently drop a hook. A hook that also touches further
lines marks each of them with ``rca-site:H<n>``; their number per file is
pinned below too.

H5 (the map_first wiring between PREPARE and COMPUTE) arrives with M5, so its
marker must not exist yet. H10 (M2) is the ``--annotation_prepare_only``
entry: it builds reference bundles from the samplesheet rows before the
preflight and empties the rows, so no pipeline stage runs; it needs neither
H5 nor any PREPARE output.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCANNED_SUFFIXES = {".nf", ".config", ".groovy", ".py"}
MARKER_PATTERN = re.compile(r"\brca-(hook|site):(H\d+)\b")

MAIN_NF = "workflows/main.nf"
# Hook -> files that carry its marker exactly once (plan §2.4 table).
EXPECTED_HOOKS: dict[str, tuple[str, ...]] = {
    "H1": (MAIN_NF,),
    "H2": (MAIN_NF,),
    "H3": (MAIN_NF,),
    "H4": (MAIN_NF,),
    "H5": (),  # M5: CLUSTERING_MAP_FIRST between PREPARE and FINALIZE.
    "H6": (MAIN_NF,),
    "H7": ("workflows/nextflow.config", "workflows/conf/dwight.config"),
    "H8": ("src/merxen/config.py",),
    "H9": ("src/merxen/io/samplesheet.py",),
    "H10": (MAIN_NF,),  # M2: --annotation_prepare_only entry.
}
# (file, hook) -> number of extra lines the hook touches, each with a site marker.
EXPECTED_SITES: dict[tuple[str, str], int] = {
    # The MENDER preflight and MENDER input key lookups, and the MENDER input
    # closure parameter they need (renamed from ``_settings``).
    (MAIN_NF, "H2"): 3,
    ("src/merxen/config.py", "H8"): 3,  # sample config, clustering config, MENDER
    ("src/merxen/io/samplesheet.py", "H9"): 2,  # SamplePair fields, row parsing
}
# Hook -> text that must follow its marker within HOOK_WINDOW lines.
HOOK_CONTENT: dict[str, tuple[str, ...]] = {
    "H1": (
        "include { ANNOTATION_PREPARE_ONLY } from "
        '"./subworkflows/annotation_references"',
        'include { CLUSTERING_MAP_FIRST } from "./subworkflows/clustering_map_first"',
    ),
    "H2": (
        'def clusteredSpatialdataTableKey(sourceTableKey, segmentation, suffix = "")',
    ),
    "H3": (
        "def annotationSettings = AnnotationSettings.forRow(row, params, species)",
        "runMapMyCells = AnnotationSettings.runMapMyCells(",
        "return annotationSettings + [",
    ),
    "H4": (
        "AnnotationPreflight.append(errors, settings, params)",
        "if (!AnnotationSettings.isLegacy(settings)) {",
    ),
    "H6": (".onComplete {", "AnnotationSettings.completionSummary("),
    "H8": ("from merxen.annotation.config import",),
    "H9": ("parse_optional_columns",),
    "H10": (
        "if (AnnotationReferences.prepareOnly(params)) {",
        "ANNOTATION_PREPARE_ONLY(sample_rows_raw_ch)",
        "sample_rows_raw_ch = channel.empty()",
    ),
}
HOOK_WINDOW = 12


def _scanned_files() -> list[Path]:
    roots = [REPO_ROOT / "workflows", REPO_ROOT / "src"]
    return sorted(
        path
        for root in roots
        for path in root.rglob("*")
        if path.is_file() and path.suffix in SCANNED_SUFFIXES
    )


def _markers() -> Counter[tuple[str, str, str]]:
    """Count ``(kind, hook, relative path)`` over the scanned files."""
    counts: Counter[tuple[str, str, str]] = Counter()
    for path in _scanned_files():
        relative = str(path.relative_to(REPO_ROOT))
        for match in MARKER_PATTERN.finditer(path.read_text()):
            counts[(match.group(1), match.group(2), relative)] += 1
    return counts


def _marker_window(relative: str, hook: str) -> str:
    lines = (REPO_ROOT / relative).read_text().splitlines()
    index = next(i for i, line in enumerate(lines) if f"rca-hook:{hook}" in line)
    return "\n".join(lines[index : index + HOOK_WINDOW])


def test_every_hook_marker_occurs_exactly_once_where_planned() -> None:
    """Each hook marker is in its planned files, once each, and nowhere else."""
    counts = _markers()
    found: dict[str, dict[str, int]] = {}
    for (kind, hook, relative), count in counts.items():
        if kind == "hook":
            found.setdefault(hook, {})[relative] = count

    for hook, files in EXPECTED_HOOKS.items():
        assert found.get(hook, {}) == dict.fromkeys(files, 1), (
            f"rca-hook:{hook} must occur exactly once in each of {list(files)}; "
            f"found {found.get(hook, {})}. Restore the hook (plan §2.4) instead "
            "of deleting or duplicating its marker."
        )
    assert set(found) <= set(EXPECTED_HOOKS), f"unknown hooks: {sorted(found)}"


def test_hook_sites_are_pinned() -> None:
    """Extra lines of a hook keep their site markers, next to the hook itself."""
    counts = _markers()
    sites = {
        (relative, hook): count
        for (kind, hook, relative), count in counts.items()
        if kind == "site"
    }

    assert sites == EXPECTED_SITES
    for relative, hook in sites:
        assert relative in EXPECTED_HOOKS[hook], (relative, hook)


def test_h5_is_left_for_m5() -> None:
    """M1 does not touch the PREPARE -> COMPUTE wiring (hook H5)."""
    main_text = (REPO_ROOT / MAIN_NF).read_text()

    assert "CLUSTERING_MAP_FIRST(" not in main_text
    assert "rca-hook:H5" not in main_text


@pytest.mark.parametrize("hook", sorted(HOOK_CONTENT))
def test_hook_marker_sits_on_its_change(hook: str) -> None:
    """A marker moved away from its code would no longer guard it."""
    (relative,) = EXPECTED_HOOKS[hook]
    window = _marker_window(relative, hook)

    for expected in HOOK_CONTENT[hook]:
        assert expected in window, f"{hook}: {expected!r} not near its marker"


def test_h10_runs_before_the_preflight_and_only_for_prepare_only_runs() -> None:
    """H10 sits between the row settings and the preflight, behind its flag.

    Emptying the rows there means a prepare-only run neither preflights nor
    runs any pipeline stage, and the prepare-only workflow is called nowhere
    else, so legacy and map_first runs never instantiate it.
    """
    main_text = (REPO_ROOT / MAIN_NF).read_text()
    hook = main_text.index("rca-hook:H10")

    assert main_text.index("sample_rows_raw_ch = samplesheet_ch.map") < hook
    assert hook < main_text.index("preflight_done_ch = sample_rows_raw_ch")
    assert main_text.count("ANNOTATION_PREPARE_ONLY(") == 1
    guard = main_text.index("if (AnnotationReferences.prepareOnly(params)) {")
    block_end = main_text.index("\n    }\n", guard)
    assert guard < main_text.index("ANNOTATION_PREPARE_ONLY(") < block_end
    assert guard < main_text.index("sample_rows_raw_ch = channel.empty()") < block_end
    for name in (
        "ANNOTATE_PANEL(",
        "ANNOTATE_REFERENCE_PREP(",
        "ANNOTATION_REFERENCES(",
        "ANNOTATION_BUNDLES(",
        "ANNOTATION_PREPARED_REFERENCES(",
    ):
        assert name not in main_text, name


def test_h4_guards_the_legacy_clustering_checks() -> None:
    """H4 opens the preflight function, before any legacy check."""
    main_text = (REPO_ROOT / MAIN_NF).read_text()
    start = main_text.index(
        "def appendClusteringSquidpyPreflightChecks(errors, settings, params) {"
    )
    body = main_text[start : main_text.index("\ndef ", start + 1)]

    assert body.index("rca-hook:H4") < body.index(
        "effectiveClusteringHierarchicalEnabled"
    )


def test_h7_includes_follow_their_markers() -> None:
    """Both configs include the annotation config files at hook H7."""
    expected = {
        "workflows/nextflow.config": "includeConfig 'conf/annotation.config'",
        "workflows/conf/dwight.config": "includeConfig 'dwight.annotation.config'",
    }
    for relative, include in expected.items():
        text = (REPO_ROOT / relative).read_text()
        assert text.count(include) == 1
        assert include in _marker_window(relative, "H7")


H2_SUFFIX_ARGUMENT = "settings.clustering_squidpy_table_key_suffix,"
H2_CLOSURE_BINDING = re.compile(r"^\s*settings, // rca-site:H2\b")


def test_h2_sites_pass_the_row_suffix() -> None:
    """Each H2 site passes the row suffix to the key function or binds it."""
    lines = (REPO_ROOT / MAIN_NF).read_text().splitlines()
    site_lines = [line for line in lines if "rca-site:H2" in line]

    suffix_lines = [line for line in site_lines if H2_SUFFIX_ARGUMENT in line]
    binding_lines = [line for line in site_lines if H2_CLOSURE_BINDING.match(line)]
    assert len(suffix_lines) == 2
    assert len(binding_lines) == 1
    assert len(site_lines) == len(suffix_lines) + len(binding_lines)


def test_mender_input_closure_binds_the_row_settings() -> None:
    """The MENDER input closure names ``settings``, which its H2 site reads.

    A rebase that restored the old ``_settings`` name would leave the key
    lookup reading an undefined variable at run time.
    """
    main_text = (REPO_ROOT / MAIN_NF).read_text()
    start = main_text.index("mender_inputs_ch = mender_artifacts_ch")
    closure = main_text[start : main_text.index("->", start)]
    body = main_text[start : main_text.index("tuple(", start)]

    parameters = [
        part.split("//")[0].strip() for part in closure.split("{", 1)[1].split("\n")
    ]
    assert "settings," in parameters
    assert "_settings," not in parameters
    assert H2_SUFFIX_ARGUMENT in body


def test_lib_classes_used_by_the_hooks_exist() -> None:
    """The hooks call classes that Nextflow loads from ``workflows/lib``."""
    lib_dir = REPO_ROOT / "workflows" / "lib"
    main_text = (REPO_ROOT / MAIN_NF).read_text()

    for name in ("AnnotationDefaults", "AnnotationSettings", "AnnotationPreflight"):
        source = (lib_dir / f"{name}.groovy").read_text()
        assert re.search(rf"^class {name} \{{", source, re.MULTILINE), name
    for name in re.findall(r"\b(Annotation[A-Z]\w+)\.\w+\(", main_text):
        assert (lib_dir / f"{name}.groovy").is_file(), name
