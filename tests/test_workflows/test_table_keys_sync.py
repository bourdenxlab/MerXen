"""Keep the SpatialData table keys in ``main.nf`` and Python equal (plan §4.8).

``workflows/main.nf`` names the tables with ``analysisLayerKeys`` and
``clusteredSpatialdataTableKey``; ``merxen.table_keys`` mirrors both. These
tests read the two Groovy functions from the source text, evaluate them on
every analysis branch and compare with the Python mapping. They also fail
while any other workflow or Python code builds a ``*_clustering_squidpy`` key
itself, so a new copy (such as the one ``feature/gaston`` adds) cannot drift.

Hook H2 gives ``clusteredSpatialdataTableKey`` a ``suffix`` parameter
(``_<suffix>`` on every return path, nothing for a null or blank suffix), and
its callers pass the row's ``clustering_squidpy_table_key_suffix`` (absent, so
null, in legacy rows); the tests evaluate ``main.nf`` with and without one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from merxen.analysis.clustering_squidpy import _clustered_spatialdata_table_key
from merxen.config import CorticalDepthTableConfig
from merxen.cortical_depth.pipeline import _clustering_table_key
from merxen.table_keys import (
    ANALYSIS_LAYER_KEYS,
    PLATFORMS,
    AnalysisLayerKeys,
    analysis_layer_keys,
    clustered_table_key,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN_NF = REPO_ROOT / "workflows" / "main.nf"
PACKAGE_ROOT = REPO_ROOT / "src" / "merxen"

# A string that builds a clustered table key: a literal ``table_*`` prefix, an
# interpolated prefix (``${...}`` / ``{...}``) or a bare ``"_clustering_squidpy"``.
# ``merxen_clustering_squidpy`` (uns key) and ``run_clustering_squidpy`` do not
# match.
KEY_BUILDER_PATTERN = re.compile(r"(?:\btable_\w*|\}|[\"'])_clustering_squidpy")

# The two statements of hook H2 that turn ``suffix`` into ``suffixFragment``,
# which every return template ends with.
SUFFIX_PREAMBLE = (
    'def suffixToken = suffix == null ? "" : suffix.toString().trim()',
    'def suffixFragment = suffixToken ? "_${suffixToken}" : ""',
)
# Suffixes the pipeline can pass: none, the pre-flip default and an explicit
# token (surrounding spaces are trimmed on both sides).
SUFFIX_CASES: tuple[str, ...] = ("", "mapfirst", " trial_2 ")

# (source table key, segmentation) pairs outside the analysisLayerKeys
# combinations: each return path is reached through the segmentation alone,
# the source key alone, and neither.
EXTRA_KEY_CASES: tuple[tuple[str, str | None], ...] = (
    ("table_custom", "reseg"),
    ("table_custom", "original_seg"),
    ("table_custom", "proseg_hybrid"),
    ("table_MOSAIK_proseg", "proseg_hybrid"),
    ("table_original", "cellpose"),
    ("table_MOSAIK_cellpose", "proseg_mask"),
    ("table", "unknown_branch"),
    ("table_custom", None),
    ("table_MOSAIK_proseg", None),
)


def _function_body(text: str, name: str) -> tuple[str, str]:
    """Return the parameter list and body of a top-level Groovy function."""
    match = re.search(rf"^def {name}\((?P<params>[^)]*)\)\s*\{{", text, re.MULTILINE)
    assert match is not None, f"{name} not found in main.nf"
    depth = 1
    index = match.end()
    while depth:
        char = text[index]
        depth += {"{": 1, "}": -1}.get(char, 0)
        index += 1
    return match.group("params"), text[match.end() : index - 1]


def _strings(text: str) -> list[str]:
    return re.findall(r'"([^"]*)"', text)


def _main_nf_analysis_layer_keys() -> dict[str, dict[str, AnalysisLayerKeys]]:
    """Evaluate ``analysisLayerKeys`` for every segmentation it names."""
    params, body = _function_body(MAIN_NF.read_text(), "analysisLayerKeys")
    assert [part.strip() for part in params.split(",")] == ["platform", "segmentation"]
    block_pattern = re.compile(
        r"if \((?P<condition>[^)]*)\)\s*\{\s*return\s*\[(?P<value>[^\]]*)\]\s*\}",
        re.DOTALL,
    )
    blocks = list(block_pattern.finditer(body))
    assert blocks, "no branches parsed from analysisLayerKeys"
    remainder = block_pattern.sub("", body).strip()
    assert remainder.startswith("throw new IllegalArgumentException"), remainder

    mapping: dict[str, dict[str, AnalysisLayerKeys]] = {}
    for block in blocks:
        condition = block.group("condition").strip()
        if condition.startswith("segmentation in ["):
            segmentations = _strings(condition)
        else:
            equals = re.fullmatch(r'segmentation == "([^"]+)"', condition)
            assert equals is not None, f"unparsed condition: {condition}"
            segmentations = [equals.group(1)]
        value = block.group("value")
        table_match = re.search(r'table_key:\s*"([^"]+)"', value)
        assert table_match is not None, value
        shape_literal = re.search(r'shape_key:\s*"([^"]+)"', value)
        shape_ternary = re.search(
            r'shape_key:\s*platform == "(?P<platform>[A-Z]+)"\s*'
            r'\?\s*"(?P<when_true>[^"]+)"\s*:\s*"(?P<when_false>[^"]+)"',
            value,
        )
        per_platform: dict[str, AnalysisLayerKeys] = {}
        for platform in PLATFORMS:
            if shape_literal is not None:
                shape_key = shape_literal.group(1)
            else:
                assert shape_ternary is not None, f"unparsed shape_key: {value}"
                shape_key = (
                    shape_ternary.group("when_true")
                    if platform == shape_ternary.group("platform")
                    else shape_ternary.group("when_false")
                )
            per_platform[platform] = AnalysisLayerKeys(table_match.group(1), shape_key)
        for segmentation in segmentations:
            assert segmentation not in mapping, f"duplicate branch {segmentation}"
            mapping[segmentation] = per_platform
    return mapping


def _suffix_fragment(suffix: str | None) -> str:
    """Evaluate ``SUFFIX_PREAMBLE`` (Groovy truth: an empty string is false)."""
    token = "" if suffix is None else suffix.strip()
    return f"_{token}" if token else ""


@dataclass(frozen=True)
class _GroovyKeyFunction:
    """``clusteredSpatialdataTableKey`` parsed into ordered return rules."""

    params: list[str]
    preamble: list[str]
    rules: list[tuple[list[tuple[str, str]], str]]
    default_template: str

    def __call__(
        self,
        source_table_key: str,
        segmentation: str | None,
        suffix: str | None = "",
    ) -> str:
        variables = {
            "sourceTableKey": source_table_key,
            "segmentation": segmentation,
            "suffixFragment": _suffix_fragment(suffix),
        }
        for clauses, template in self.rules:
            if any(variables[name] == value for name, value in clauses):
                return self._render(template, variables)
        return self._render(self.default_template, variables)

    @staticmethod
    def _render(template: str, variables: dict[str, str | None]) -> str:
        def substitute(match: re.Match[str]) -> str:
            value = variables[match.group(1)]
            assert value is not None, match.group(0)
            return value

        return re.sub(r"\$\{(\w+)\}", substitute, template)


def _main_nf_clustered_table_key() -> _GroovyKeyFunction:
    params, body = _function_body(MAIN_NF.read_text(), "clusteredSpatialdataTableKey")
    rule_pattern = re.compile(
        r'if \((?P<condition>[^)]*)\)\s*\{\s*return\s+"(?P<template>[^"]*)"\s*\}'
    )
    rules: list[tuple[list[tuple[str, str]], str]] = []
    for match in rule_pattern.finditer(body):
        clauses = []
        for clause in match.group("condition").split("||"):
            parsed = re.fullmatch(r'\s*(\w+) == "([^"]*)"\s*', clause)
            assert parsed is not None, f"unparsed clause: {clause}"
            clauses.append((parsed.group(1), parsed.group(2)))
        rules.append((clauses, match.group("template")))
    remainder = [
        line.strip() for line in rule_pattern.sub("", body).splitlines() if line.strip()
    ]
    preamble = [line for line in remainder if line.startswith("def ")]
    default = re.fullmatch(
        r'return\s+"(?P<template>[^"]*)"', " ".join(remainder[len(preamble) :])
    )
    assert default is not None, "unparsed default return of the key function"
    return _GroovyKeyFunction(
        params=[part.strip() for part in params.split(",")],
        preamble=preamble,
        rules=rules,
        default_template=default.group("template"),
    )


def _key_cases() -> list[tuple[str, str | None]]:
    cases: list[tuple[str, str | None]] = [
        (keys.table_key, segmentation)
        for segmentation, per_platform in ANALYSIS_LAYER_KEYS.items()
        for keys in per_platform.values()
    ]
    cases.extend(EXTRA_KEY_CASES)
    return sorted(set(cases), key=str)


KEY_CASES = _key_cases()


def test_analysis_layer_keys_match_main_nf() -> None:
    """Every analysisLayerKeys branch and alias has the same keys in Python."""
    expected = _main_nf_analysis_layer_keys()

    assert set(ANALYSIS_LAYER_KEYS) == set(expected)
    for segmentation, per_platform in expected.items():
        for platform, keys in per_platform.items():
            assert ANALYSIS_LAYER_KEYS[segmentation][platform] == keys
            assert analysis_layer_keys(platform, segmentation) == keys


def test_main_nf_clustered_key_function_has_three_return_paths() -> None:
    """Reseg, original_seg and the default path are the only returns."""
    function = _main_nf_clustered_table_key()

    assert function.params == ["sourceTableKey", "segmentation", 'suffix = ""']
    assert function.preamble == list(SUFFIX_PREAMBLE)
    assert [clauses for clauses, _ in function.rules] == [
        [("segmentation", "reseg"), ("sourceTableKey", "table_MOSAIK_proseg")],
        [("segmentation", "original_seg"), ("sourceTableKey", "table_original")],
    ]
    assert function.default_template.startswith("${sourceTableKey}")
    for template in [template for _, template in function.rules] + [
        function.default_template
    ]:
        assert template.endswith("_clustering_squidpy${suffixFragment}"), template


@pytest.mark.parametrize(("source_table_key", "segmentation"), KEY_CASES)
def test_clustered_table_key_matches_main_nf(
    source_table_key: str, segmentation: str | None
) -> None:
    """The unsuffixed Python key equals main.nf on every return path."""
    main_nf_key = _main_nf_clustered_table_key()(source_table_key, segmentation)

    assert clustered_table_key(source_table_key, segmentation) == main_nf_key
    assert clustered_table_key(source_table_key, segmentation, "") == main_nf_key


@pytest.mark.parametrize("suffix", SUFFIX_CASES)
@pytest.mark.parametrize(("source_table_key", "segmentation"), KEY_CASES)
def test_suffixed_clustered_table_key_matches_main_nf(
    source_table_key: str, segmentation: str | None, suffix: str
) -> None:
    """Hook H2: the suffixed Python key equals main.nf on every return path."""
    main_nf_key = _main_nf_clustered_table_key()(source_table_key, segmentation, suffix)

    assert clustered_table_key(source_table_key, segmentation, suffix) == main_nf_key


@pytest.mark.parametrize(("source_table_key", "segmentation"), KEY_CASES)
def test_main_nf_null_suffix_keeps_the_legacy_key(
    source_table_key: str, segmentation: str | None
) -> None:
    """Legacy rows have no suffix setting, so main.nf gets null: no suffix."""
    function = _main_nf_clustered_table_key()

    assert function(source_table_key, segmentation, None) == clustered_table_key(
        source_table_key, segmentation
    )
    assert function(source_table_key, segmentation, "  ") == function(
        source_table_key, segmentation
    )


def test_main_nf_callers_pass_the_row_suffix() -> None:
    """Every caller of the key function passes the row's table-key suffix."""
    main_text = MAIN_NF.read_text()
    calls = [
        match
        for match in re.finditer(r"clusteredSpatialdataTableKey\(([^)]*)\)", main_text)
        if not main_text[: match.start()].endswith("def ")
    ]

    assert len(calls) >= 2
    for call in calls:
        code = re.sub(r"//[^\n]*", "", call.group(1))
        arguments = [part.strip() for part in code.split(",") if part.strip()]
        assert len(arguments) == 3, call.group(0)
        assert arguments[2].startswith("settings.clustering_squidpy_table_key_suffix")


@pytest.mark.parametrize(("source_table_key", "segmentation"), KEY_CASES)
def test_clustered_table_key_suffix_applies_on_every_path(
    source_table_key: str, segmentation: str | None
) -> None:
    """A suffix appends ``_<suffix>`` to the unsuffixed key on every path."""
    unsuffixed = clustered_table_key(source_table_key, segmentation)

    assert unsuffixed.endswith("_clustering_squidpy")
    assert (
        clustered_table_key(source_table_key, segmentation, "mapfirst")
        == f"{unsuffixed}_mapfirst"
    )
    assert (
        clustered_table_key(source_table_key, segmentation, " trial_2 ")
        == f"{unsuffixed}_trial_2"
    )


def test_clustered_table_key_return_paths() -> None:
    """The three return paths give the documented keys."""
    assert (
        clustered_table_key("table_custom", "reseg")
        == "table_MOSAIK_proseg_clustering_squidpy"
    )
    assert (
        clustered_table_key("table_MOSAIK_proseg", None, "mapfirst")
        == "table_MOSAIK_proseg_clustering_squidpy_mapfirst"
    )
    assert (
        clustered_table_key("table_custom", "Original_Seg")
        == "table_original_clustering_squidpy"
    )
    assert (
        clustered_table_key("table_MOSAIK_proseg_hybrid", "proseg_hybrid", "mapfirst")
        == "table_MOSAIK_proseg_hybrid_clustering_squidpy_mapfirst"
    )


@pytest.mark.parametrize("suffix", ["Mapfirst", "map/first", "map first", "-x"])
def test_clustered_table_key_rejects_invalid_suffix(suffix: str) -> None:
    """A suffix must be a lower-case token, so the zarr key stays valid."""
    with pytest.raises(ValueError, match="lower-case token"):
        clustered_table_key("table_MOSAIK_proseg", "reseg", suffix)


def test_analysis_layer_keys_rejects_unknown_branch() -> None:
    """Unknown segmentations and platforms fail, as main.nf throws."""
    with pytest.raises(ValueError, match="Unknown analysis segmentation"):
        analysis_layer_keys("XENIUM", "watershed")
    with pytest.raises(ValueError, match="Unknown platform"):
        analysis_layer_keys("VISIUM", "reseg")
    assert analysis_layer_keys(" merscope ", "Original") == AnalysisLayerKeys(
        "table_original", "merscope_cell_boundaries"
    )


@pytest.mark.parametrize(("source_table_key", "segmentation"), KEY_CASES)
def test_python_consumers_delegate_to_table_keys(
    source_table_key: str, segmentation: str | None
) -> None:
    """The clustering writer and cortical depth use the shared mapping."""
    expected = clustered_table_key(source_table_key, segmentation)

    assert _clustered_spatialdata_table_key(source_table_key, segmentation) == expected
    if segmentation is not None:
        table_config = CorticalDepthTableConfig(
            segmentation=segmentation, table_key=source_table_key
        )
        assert _clustering_table_key(table_config) == expected


def test_only_clustered_spatialdata_table_key_builds_keys_in_workflows() -> None:
    """No other workflow code builds a ``*_clustering_squidpy`` key."""
    main_text = MAIN_NF.read_text()
    _, body = _function_body(main_text, "clusteredSpatialdataTableKey")
    in_function = len(KEY_BUILDER_PATTERN.findall(body))

    assert in_function >= 3
    assert len(KEY_BUILDER_PATTERN.findall(main_text)) == in_function

    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in sorted((REPO_ROOT / "workflows").rglob("*"))
        if path.is_file()
        and path != MAIN_NF
        and path.suffix in {".nf", ".groovy", ".config"}
        and KEY_BUILDER_PATTERN.search(path.read_text())
    ]
    assert offenders == []


def test_only_table_keys_builds_keys_in_python() -> None:
    """No other package module builds a ``*_clustering_squidpy`` key."""
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in sorted(PACKAGE_ROOT.rglob("*.py"))
        if path.name != "table_keys.py" and KEY_BUILDER_PATTERN.search(path.read_text())
    ]
    assert offenders == []
