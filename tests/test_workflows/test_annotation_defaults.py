"""Annotation defaults in Groovy equal the Python ones (plan §3.7, §4.8).

``workflows/lib/AnnotationDefaults.groovy`` decides the clustering mode and the
clustered table-key suffix of a run in Nextflow; ``merxen.annotation.config``
decides them in Python. The structural tests compare the Groovy constants,
``forSpecies`` and the ``conf/annotation.config`` defaults with the Python
side. When ``nextflow`` is installed, the Groovy classes also run in a tiny
Nextflow script, covering ``resolveMode`` and ``tableKeySuffix`` in all four
mode x flip combinations, ``AnnotationSettings`` and ``AnnotationPreflight``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from merxen.annotation import config as annotation_config
from merxen.annotation.config import (
    CLUSTERING_MODES,
    DEFAULT_CLUSTERING_MODE,
    DEFAULT_HUMAN_REGION,
    DEFAULT_REFERENCE_IDS,
    FLIPPED_SPECIES,
    KNOWN_REFERENCES,
    MAP_FIRST_TABLE_KEY_SUFFIX,
    MAP_FIRST_UNASSIGNED_STATE_POLICY,
    MOUSE_CCF_REGIONS,
    MOUSE_SECTION_REGION_KEYWORDS,
    VALIDATED_HUMAN_REGIONS,
    AnnotationConfig,
    AnnotationReferenceSpec,
    resolve_clustering_mode,
    resolve_table_key_suffix,
    species_defaults,
)
from merxen.annotation.vocab import SPECIES
from merxen.table_keys import TABLE_KEY_SUFFIX_PATTERN

REPO_ROOT = Path(__file__).resolve().parents[2]
LIB_DIR = REPO_ROOT / "workflows" / "lib"
DEFAULTS_SOURCE = (LIB_DIR / "AnnotationDefaults.groovy").read_text()
SETTINGS_SOURCE = (LIB_DIR / "AnnotationSettings.groovy").read_text()
NEXTFLOW = shutil.which("nextflow")
needs_nextflow = pytest.mark.skipif(NEXTFLOW is None, reason="nextflow is unavailable")

# --------------------------------------------------------------------------
# Groovy source parsing


def _groovy_constant(source: str, name: str) -> str:
    """Return the initializer text of ``static final ... NAME = ...``."""
    match = re.search(rf"static final [\w<>, ]+ {name} = ", source)
    assert match is not None, f"{name} not found"
    start = match.end()
    if source[start] == "[":
        depth = 0
        for index in range(start, len(source)):
            depth += {"[": 1, "]": -1}.get(source[index], 0)
            if depth == 0:
                return source[start : index + 1]
    return source[start : source.index("\n", start)]


def _groovy_strings(text: str) -> list[str]:
    return re.findall(r'"([^"]*)"', text)


def _groovy_string_constant(source: str, name: str) -> str:
    text = _groovy_constant(source, name)
    match = re.fullmatch(r"""["'](.*)["']""", text.strip())
    assert match is not None, text
    return match.group(1)


def _groovy_species_lists(source: str, name: str) -> dict[str, list[str]]:
    text = _groovy_constant(source, name)
    return {
        species: _groovy_strings(
            re.search(rf"{species}: \[(.*?)\]", text, re.DOTALL).group(1)  # type: ignore[union-attr]
        )
        for species in SPECIES
    }


def _groovy_literal(text: str) -> Any:
    """Convert a scalar Groovy literal (string, boolean, null, number)."""
    text = text.strip()
    if text.startswith('"'):
        return text[1 : text.index('"', 1)]
    token = text.split("//")[0].strip()
    if token in {"true", "false"}:
        return token == "true"
    if token == "null":
        return None
    if re.fullmatch(r"-?\d+", token):
        return int(token)
    if re.fullmatch(r"-?\d+\.\d+", token):
        return float(token)
    raise AssertionError(f"unsupported Groovy literal: {text!r}")


def _groovy_for_species() -> dict[str, dict[str, Any]]:
    """Evaluate ``AnnotationDefaults.forSpecies`` from its source text."""
    start = DEFAULTS_SOURCE.index("static Map forSpecies(")
    body = DEFAULTS_SOURCE[start : DEFAULTS_SOURCE.index("\n    }\n", start)]
    blocks = re.findall(r"return \[\n(.*?)\n\s*\]", body, re.DOTALL)
    assert len(blocks) == 2, "forSpecies should return one map per species"
    flipped = _groovy_strings(_groovy_constant(DEFAULTS_SOURCE, "FLIPPED_SPECIES"))
    regions = _groovy_strings(
        _groovy_constant(DEFAULTS_SOURCE, "VALIDATED_HUMAN_REGIONS")
    )
    references = _groovy_species_lists(DEFAULTS_SOURCE, "DEFAULT_REFERENCES")
    result: dict[str, dict[str, Any]] = {}
    for species, block in zip(("human", "mouse"), blocks, strict=True):
        values: dict[str, Any] = {}
        for key, raw in re.findall(r"^\s*(\w+): (.+),$", block, re.MULTILINE):
            if raw == "defaultMode(speciesName)":
                values[key] = "map_first" if species in flipped else "legacy"
            elif raw == "VALIDATED_HUMAN_REGIONS[0]":
                values[key] = regions[0]
            elif raw == f"DEFAULT_REFERENCES.{species}":
                values[key] = references[species]
            else:
                values[key] = _groovy_literal(raw)
        result[species] = values
    return result


def _annotation_config_defaults(
    workflow_config_params: dict[str, dict[str, str]],
) -> dict[str, Any]:
    return {
        name: _groovy_literal(raw)
        for name, raw in workflow_config_params["conf/annotation.config"].items()
    }


# --------------------------------------------------------------------------
# Structural tests (no Nextflow needed)


def test_groovy_constants_match_python() -> None:
    """Modes, flips, suffix rules, regions and references agree."""
    source = DEFAULTS_SOURCE
    flipped = _groovy_strings(_groovy_constant(source, "FLIPPED_SPECIES"))

    assert sorted(flipped) == sorted(FLIPPED_SPECIES)
    assert tuple(_groovy_strings(_groovy_constant(source, "SPECIES"))) == SPECIES
    assert _groovy_constant(source, "MODES") == "[LEGACY, MAP_FIRST]"
    assert (
        _groovy_string_constant(source, "LEGACY"),
        _groovy_string_constant(source, "MAP_FIRST"),
    ) == CLUSTERING_MODES
    assert (
        _groovy_string_constant(source, "PRE_FLIP_TABLE_KEY_SUFFIX")
        == MAP_FIRST_TABLE_KEY_SUFFIX
    )
    assert (
        _groovy_string_constant(source, "TABLE_KEY_SUFFIX_PATTERN")
        == TABLE_KEY_SUFFIX_PATTERN.pattern
    )
    assert (
        tuple(_groovy_strings(_groovy_constant(source, "VALIDATED_HUMAN_REGIONS")))
        == VALIDATED_HUMAN_REGIONS
    )
    assert (
        tuple(
            _groovy_strings(_groovy_constant(source, "MOUSE_SECTION_REGION_KEYWORDS"))
        )
        == MOUSE_SECTION_REGION_KEYWORDS
    )
    assert (
        tuple(_groovy_strings(_groovy_constant(source, "MOUSE_CCF_REGIONS")))
        == MOUSE_CCF_REGIONS
    )
    assert _groovy_species_lists(source, "DEFAULT_REFERENCES") == {
        species: list(ids) for species, ids in DEFAULT_REFERENCE_IDS.items()
    }
    known = _groovy_species_lists(source, "KNOWN_REFERENCES")
    assert {species: sorted(ids) for species, ids in known.items()} == {
        species: sorted(
            reference_id
            for reference_id, spec in KNOWN_REFERENCES.items()
            if spec["species"] == species
        )
        for species in SPECIES
    }


def test_python_default_modes_follow_the_flips() -> None:
    """A species defaults to map_first exactly when it has flipped."""
    for species in SPECIES:
        expected = "map_first" if species in FLIPPED_SPECIES else "legacy"
        assert DEFAULT_CLUSTERING_MODE[species] == expected


@pytest.mark.parametrize("species", SPECIES)
def test_groovy_for_species_matches_species_defaults(species: str) -> None:
    """``forSpecies`` returns the pydantic ``species_defaults``."""
    assert _groovy_for_species()[species] == species_defaults(species)  # type: ignore[arg-type]


def test_annotation_config_defaults_match_the_models(
    workflow_config_params: dict[str, dict[str, str]],
) -> None:
    """The Nextflow defaults of annotation.config equal the pydantic defaults."""
    defaults = _annotation_config_defaults(workflow_config_params)
    model = AnnotationConfig()
    expected = {
        "clustering_squidpy_mode_human": DEFAULT_CLUSTERING_MODE["human"],
        "clustering_squidpy_mode_mouse": DEFAULT_CLUSTERING_MODE["mouse"],
        "clustering_squidpy_mode": None,
        "clustering_squidpy_table_key_suffix": None,
        "annotation_human_references": ",".join(DEFAULT_REFERENCE_IDS["human"]),
        "annotation_mouse_references": ",".join(DEFAULT_REFERENCE_IDS["mouse"]),
        "annotation_human_region": DEFAULT_HUMAN_REGION,
        "annotation_mouse_section_regions": model.mouse_section_regions,
        "annotation_panel_mode": model.panel.panel_mode,
        "annotation_resolvability": model.resolvability.enabled,
        "annotation_allow_fine_levels": model.thresholds.allow_fine_levels,
        "annotation_wmb_max_cells_per_cluster": AnnotationReferenceSpec.model_fields[
            "max_cells_per_cluster"
        ].default,
        "annotation_ctm_version": model.ctm_version,
        "annotation_keep_extended_json": model.keep_extended_json,
        "annotation_reuse_published": model.reuse_published,
        "annotation_xplat_sensitivity": model.xplat_sensitivity,
        "annotation_xplat_sensitivity_segmentations": ",".join(
            model.xplat_sensitivity_segmentations
        ),
        "annotation_human_microglia_flag": species_defaults("human")[
            "microglial_spillover_enabled"
        ],
        "annotation_allow_single_method": model.allow_single_method,
        "mender_unassigned_state_policy": MAP_FIRST_UNASSIGNED_STATE_POLICY,
    }

    assert {name: defaults[name] for name in expected} == expected


@pytest.mark.parametrize("species", SPECIES)
@pytest.mark.parametrize("flipped", [False, True])
@pytest.mark.parametrize("mode", CLUSTERING_MODES)
def test_python_suffix_in_all_mode_flip_combinations(
    species: str, flipped: bool, mode: str
) -> None:
    """Legacy never gets a suffix; map_first gets "mapfirst" until the flip."""
    flips = frozenset({species}) if flipped else frozenset()
    expected = "mapfirst" if mode == "map_first" and not flipped else ""

    assert resolve_table_key_suffix(species, mode, flipped_species=flips) == expected  # type: ignore[arg-type]
    if mode == "map_first" and not flipped:
        with pytest.raises(ValueError, match="OD-A3"):
            resolve_table_key_suffix(species, mode, "", flipped_species=flips)  # type: ignore[arg-type]
    else:
        assert resolve_table_key_suffix(species, mode, "", flipped_species=flips) == ""  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Executed Groovy (Nextflow available)

HARNESS_CLASS = """
import groovy.json.JsonOutput
import groovy.json.JsonSlurper

class AnnotationTestHarness {
    static void run(Object casesPath, Object outPath) {
        Map cases = new JsonSlurper().parse(new File(casesPath.toString())) as Map
        Map results = [:]
        cases.each { name, testCase -> results[name] = evaluate(testCase as Map) }
        new File(outPath.toString()).text = JsonOutput.toJson(results)
    }

    static Map evaluate(Map testCase) {
        try {
            return [value: call(testCase)]
        } catch (IllegalArgumentException error) {
            return [error: error.message]
        }
    }

    static Object call(Map c) {
        def flipped = c.flipped == null ? AnnotationDefaults.FLIPPED_SPECIES : c.flipped
        switch (c.fn) {
            case "resolveMode":
                return AnnotationDefaults.resolveMode(c.params, c.species, flipped)
            case "tableKeySuffix":
                return AnnotationDefaults.tableKeySuffix(c.params, c.species, flipped)
            case "forSpecies":
                return AnnotationDefaults.forSpecies(c.species)
            case "forRow":
                return AnnotationSettings.forRow(c.row, c.params, c.species)
            case "accessors":
                return [
                    mode: AnnotationSettings.mode(c.settings),
                    legacy: AnnotationSettings.isLegacy(c.settings),
                    map_first: AnnotationSettings.isMapFirst(c.settings),
                    suffix: AnnotationSettings.tableKeySuffix(c.settings),
                ]
            case "runMapMyCells":
                return AnnotationSettings.runMapMyCells(
                    c.selected, c.settings, c.stop_stage
                )
            case "completionSummary":
                return AnnotationSettings.completionSummary(c.params, c.run_info ?: [:])
            case "preflight":
                def errors = []
                AnnotationPreflight.append(errors, c.settings, c.params)
                return errors
        }
        throw new IllegalStateException("unknown case function ${c.fn}")
    }
}
"""
HARNESS_MAIN = (
    "workflow {\n    AnnotationTestHarness.run(params.cases, params.out)\n}\n"
)


def _mode_params(
    species: str, source: str, mode: str | None, suffix: str | None = None
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "clustering_squidpy_mode_human": "legacy",
        "clustering_squidpy_mode_mouse": "legacy",
        "clustering_squidpy_mode": None,
        "clustering_squidpy_table_key_suffix": suffix,
    }
    if source == "override":
        params["clustering_squidpy_mode"] = mode
    elif source == "species_param":
        params[f"clustering_squidpy_mode_{species}"] = mode
    else:
        params[f"clustering_squidpy_mode_{species}"] = None
    return params


MODE_MATRIX = [
    (species, flipped, source, mode)
    for species in SPECIES
    for flipped in (False, True)
    for source, mode in (
        ("override", "legacy"),
        ("override", "map_first"),
        ("species_param", "legacy"),
        ("species_param", "map_first"),
        ("default", None),
    )
]


# Mode parsing: label -> (params, species alias for Groovy, Python species).
MODE_PARSING_CASES: dict[str, tuple[dict[str, Any], str, str]] = {
    "bad-mode": ({"clustering_squidpy_mode": "map-first"}, "human", "human"),
    "case-and-space": (
        {"clustering_squidpy_mode_mouse": " MAP_FIRST "},
        "Mus musculus",
        "mouse",
    ),
    "blank-override": (
        {"clustering_squidpy_mode": " ", "clustering_squidpy_mode_human": "map_first"},
        "homo_sapiens",
        "human",
    ),
    "mixed-case-override": (
        {
            "clustering_squidpy_mode": "Legacy",
            "clustering_squidpy_mode_human": "map_first",
        },
        "human",
        "human",
    ),
    "blank-species-param": ({"clustering_squidpy_mode_mouse": "  "}, "mouse", "mouse"),
}


def _case_name(species: str, flipped: bool, source: str, mode: str | None) -> str:
    return f"{species}|{'flipped' if flipped else 'unflipped'}|{source}|{mode}"


def _build_cases(tmp_path: Path, defaults: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for species, flipped, source, mode in MODE_MATRIX:
        flips = [species] if flipped else []
        params = _mode_params(species, source, mode)
        name = _case_name(species, flipped, source, mode)
        for fn in ("resolveMode", "tableKeySuffix"):
            cases[f"{fn}|{name}"] = {
                "fn": fn,
                "params": params,
                "species": species,
                "flipped": flips,
            }
        for suffix in ("trial_2", "", "Bad/Key"):
            cases[f"tableKeySuffix|{name}|suffix={suffix}"] = {
                "fn": "tableKeySuffix",
                "params": _mode_params(species, source, mode, suffix),
                "species": species,
                "flipped": flips,
            }
    for label, (params, species_alias, _species) in MODE_PARSING_CASES.items():
        cases[f"resolveMode|{label}"] = {
            "fn": "resolveMode",
            "params": params,
            "species": species_alias,
        }
    cases["resolveMode|bad-species"] = {
        "fn": "resolveMode",
        "params": {},
        "species": "zebrafish",
    }
    for species in SPECIES:
        cases[f"forSpecies|{species}"] = {"fn": "forSpecies", "species": species}
        cases[f"resolveMode|config-defaults|{species}"] = {
            "fn": "resolveMode",
            "params": defaults,
            "species": species,
        }
        cases[f"forRow|config-defaults|{species}"] = {
            "fn": "forRow",
            "row": {
                "pair_id": "P1",
                "anatomical_region": "x",
                "mouse_section_regions": "x",
            },
            "params": defaults,
            "species": species,
        }

    map_first = {**defaults, "clustering_squidpy_mode": "map_first"}
    cases["forRow|human|row-region"] = {
        "fn": "forRow",
        "row": {"pair_id": "P1", "anatomical_region": " Frontal-Cortex "},
        "params": map_first,
        "species": "human",
    }
    cases["forRow|human|param-region"] = {
        "fn": "forRow",
        "row": {"pair_id": "P1", "anatomical_region": "  "},
        "params": map_first,
        "species": "human",
    }
    cases["forRow|mouse|row-regions"] = {
        "fn": "forRow",
        "row": {"pair_id": "M1", "mouse_section_regions": "Isocortex;HPF"},
        "params": {**map_first, "clustering_squidpy_table_key_suffix": "trial"},
        "species": "mouse",
    }
    cases["forRow|mouse|param-regions"] = {
        "fn": "forRow",
        "row": {"pair_id": "M1"},
        "params": map_first,
        "species": "mouse",
    }

    map_first_settings = {
        "clustering_squidpy_mode": "map_first",
        "clustering_squidpy_table_key_suffix": "mapfirst",
        "annotation_mode_mapmycells_stage": "legacy",
    }
    cases["accessors|legacy"] = {"fn": "accessors", "settings": {"pair_id": "P1"}}
    cases["accessors|map_first"] = {"fn": "accessors", "settings": map_first_settings}
    for label, settings, selected, stop_stage in (
        ("legacy|selected", {}, True, "mender"),
        ("legacy|unselected", {}, False, "mapmycells"),
        ("map_first|stop-mapmycells", map_first_settings, True, "mapmycells"),
        ("map_first|stop-mender", map_first_settings, True, "mender"),
        ("map_first|unselected", map_first_settings, False, "mapmycells"),
        (
            "map_first|skip",
            {**map_first_settings, "annotation_mode_mapmycells_stage": "skip"},
            True,
            "mapmycells",
        ),
    ):
        cases[f"runMapMyCells|{label}"] = {
            "fn": "runMapMyCells",
            "settings": settings,
            "selected": selected,
            "stop_stage": stop_stage,
        }

    cases["completionSummary|legacy"] = {
        "fn": "completionSummary",
        "params": {**defaults, "species": "human"},
    }
    cases["completionSummary|bad-species"] = {
        "fn": "completionSummary",
        "params": {**defaults, "species": "zebrafish"},
    }
    cases["completionSummary|mouse-map_first"] = {
        "fn": "completionSummary",
        "params": {
            **defaults,
            "species": "mouse",
            "clustering_squidpy_mode_mouse": "map_first",
        },
        "run_info": {"success": False, "failed_annotations": ["M1:reseg"]},
    }

    existing = tmp_path / "aliases.csv"
    existing.write_text("symbol,previous\n")
    human_settings = {
        **map_first_settings,
        "species": "human",
        "pair_id": "P1",
        "annotation_anatomical_region": "frontal_cortex",
        "run_clustering_squidpy": True,
    }
    mouse_settings = {
        **map_first_settings,
        "species": "mouse",
        "pair_id": "M1",
        "annotation_mouse_section_regions": "Isocortex;hpf",
        "run_clustering_squidpy": False,
    }
    cases["preflight|legacy"] = {
        "fn": "preflight",
        "settings": {
            "species": "human",
            "pair_id": "P1",
            "run_clustering_squidpy": True,
        },
        "params": {**defaults, "annotation_panel_mode": "bogus"},
    }
    cases["preflight|map_first|clustering"] = {
        "fn": "preflight",
        "settings": human_settings,
        "params": {
            **defaults,
            "annotation_gene_alias_table": str(existing),
            # The human held-out self-map sources (resolvability on).
            "annotation_whb_region_precompute_source": str(existing),
            "annotation_whb_h5ad_dir": str(existing),
            "annotation_whb_metadata_dir": str(existing),
        },
    }
    cases["preflight|map_first|human-no-selfmap-sources"] = {
        "fn": "preflight",
        "settings": human_settings,
        "params": defaults,
    }
    cases["preflight|map_first|no-clustered-stage"] = {
        "fn": "preflight",
        "settings": mouse_settings,
        "params": defaults,
    }
    cases["preflight|map_first|mouse-no-test-cells"] = {
        "fn": "preflight",
        "settings": {**mouse_settings, "run_clustering_squidpy": True},
        "params": defaults,
    }
    cases["preflight|map_first|invalid"] = {
        "fn": "preflight",
        "settings": {
            **human_settings,
            "annotation_anatomical_region": "temporal_cortex",
            "run_clustering_squidpy": False,
        },
        "params": {
            **defaults,
            "annotation_panel_mode": "bogus",
            "annotation_human_references": "whb_frontal_supc_clus,wmb_panel",
            "annotation_ctm_version": " ",
            "annotation_gene_id_overrides_csv": str(tmp_path / "missing.csv"),
        },
    }
    cases["preflight|map_first|empty-suffix"] = {
        "fn": "preflight",
        "settings": {
            **mouse_settings,
            "clustering_squidpy_table_key_suffix": "",
        },
        "params": defaults,
    }
    cases["preflight|map_first|mouse-regions"] = {
        "fn": "preflight",
        "settings": {
            **mouse_settings,
            "annotation_mouse_section_regions": "Isocortex;Cortex",
        },
        "params": defaults,
    }
    return cases


@pytest.fixture(scope="module")
def groovy_results(
    tmp_path_factory: pytest.TempPathFactory,
    workflow_config_params: dict[str, dict[str, str]],
) -> dict[str, dict[str, Any]]:
    """Run every case through the Groovy classes in one Nextflow run."""
    assert NEXTFLOW is not None
    root = tmp_path_factory.mktemp("annotation_groovy")
    (root / "lib").mkdir()
    for source in LIB_DIR.glob("*.groovy"):
        shutil.copy(source, root / "lib" / source.name)
    (root / "lib" / "AnnotationTestHarness.groovy").write_text(HARNESS_CLASS)
    (root / "main.nf").write_text(HARNESS_MAIN)
    cases = _build_cases(root, _annotation_config_defaults(workflow_config_params))
    (root / "cases.json").write_text(json.dumps(cases))
    env = {**os.environ, "NXF_DISABLE_CHECK_LATEST": "true", "NXF_ANSI_LOG": "false"}
    completed = subprocess.run(
        [
            NEXTFLOW,
            "-log",
            str(root / "nextflow.log"),
            "run",
            "main.nf",
            "--cases",
            str(root / "cases.json"),
            "--out",
            str(root / "results.json"),
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    results: dict[str, dict[str, Any]] = json.loads((root / "results.json").read_text())
    assert set(results) == set(cases)
    return results


def _value(results: dict[str, dict[str, Any]], name: str) -> Any:
    assert "error" not in results[name], results[name]
    return results[name]["value"]


def _python_mode(
    monkeypatch: pytest.MonkeyPatch, species: str, flipped: bool, params: dict[str, Any]
) -> str:
    if flipped:
        monkeypatch.setitem(
            annotation_config.DEFAULT_CLUSTERING_MODE, species, "map_first"
        )
    return resolve_clustering_mode(
        species,  # type: ignore[arg-type]
        mode=params["clustering_squidpy_mode"],
        mode_human=params["clustering_squidpy_mode_human"],
        mode_mouse=params["clustering_squidpy_mode_mouse"],
    )


@needs_nextflow
@pytest.mark.parametrize(("species", "flipped", "source", "mode"), MODE_MATRIX)
def test_groovy_mode_and_suffix_in_all_mode_flip_combinations(
    groovy_results: dict[str, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    species: str,
    flipped: bool,
    source: str,
    mode: str | None,
) -> None:
    """resolveMode / tableKeySuffix equal Python and the plan's table."""
    name = _case_name(species, flipped, source, mode)
    expected_mode = mode or ("map_first" if flipped else "legacy")
    expected_suffix = "mapfirst" if expected_mode == "map_first" and not flipped else ""
    flips = frozenset({species}) if flipped else frozenset()
    python_mode = _python_mode(
        monkeypatch, species, flipped, _mode_params(species, source, mode)
    )

    assert _value(groovy_results, f"resolveMode|{name}") == expected_mode == python_mode
    assert (
        _value(groovy_results, f"tableKeySuffix|{name}")
        == expected_suffix
        == resolve_table_key_suffix(species, python_mode, flipped_species=flips)  # type: ignore[arg-type]
    )
    explicit = "trial_2" if expected_mode == "map_first" else ""
    assert _value(groovy_results, f"tableKeySuffix|{name}|suffix=trial_2") == explicit
    empty = groovy_results[f"tableKeySuffix|{name}|suffix="]
    if expected_mode == "map_first" and not flipped:
        # An empty suffix would overwrite the legacy table (OD-A3).
        assert "OD-A3" in empty["error"]
        with pytest.raises(ValueError, match="OD-A3"):
            resolve_table_key_suffix(species, python_mode, "", flipped_species=flips)  # type: ignore[arg-type]
    else:
        assert empty == {"value": ""}
        assert (
            resolve_table_key_suffix(species, python_mode, "", flipped_species=flips)  # type: ignore[arg-type]
            == ""
        )
    invalid = groovy_results[f"tableKeySuffix|{name}|suffix=Bad/Key"]
    if expected_mode == "map_first":
        assert "lower-case token" in invalid["error"]
    else:
        assert invalid == {"value": ""}


def _python_parsed_mode(label: str) -> dict[str, str]:
    params, _alias, species = MODE_PARSING_CASES[label]
    try:
        return {
            "value": resolve_clustering_mode(
                species,  # type: ignore[arg-type]
                mode=params.get("clustering_squidpy_mode"),
                mode_human=params.get("clustering_squidpy_mode_human"),
                mode_mouse=params.get("clustering_squidpy_mode_mouse"),
            )
        }
    except ValueError as error:
        return {"error": str(error)}


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("bad-mode", None),
        ("case-and-space", "map_first"),
        ("blank-override", "map_first"),
        ("mixed-case-override", "legacy"),
        ("blank-species-param", "legacy"),
    ],
)
def test_python_mode_parsing(label: str, expected: str | None) -> None:
    """Python normalises modes as resolveMode does (see the Groovy test)."""
    result = _python_parsed_mode(label)

    if expected is None:
        assert "clustering mode must be one of" in result["error"]
    else:
        assert result == {"value": expected}


@needs_nextflow
def test_groovy_mode_parsing(groovy_results: dict[str, dict[str, Any]]) -> None:
    """Modes are case-insensitive; blanks fall through; unknown values fail."""
    assert (
        "Unknown clustering_squidpy_mode 'map-first'"
        in (groovy_results["resolveMode|bad-mode"]["error"])
    )
    assert _value(groovy_results, "resolveMode|case-and-space") == "map_first"
    assert _value(groovy_results, "resolveMode|blank-override") == "map_first"
    assert "Unknown species" in groovy_results["resolveMode|bad-species"]["error"]
    for label in MODE_PARSING_CASES:
        groovy = groovy_results[f"resolveMode|{label}"]
        python = _python_parsed_mode(label)
        assert set(groovy) == set(python), (label, groovy, python)
        if "value" in groovy:
            assert groovy == python, label


@needs_nextflow
@pytest.mark.parametrize("species", SPECIES)
def test_groovy_config_defaults_are_legacy_and_inert(
    groovy_results: dict[str, dict[str, Any]], species: str
) -> None:
    """With annotation.config defaults both species are legacy, rows get no keys.

    VALIDATE_ANALYSIS_LAYER takes the row settings as a ``val`` input, so an
    extra key in a legacy row would change its task hash (legacy ``-resume``).
    """
    assert _value(groovy_results, f"resolveMode|config-defaults|{species}") == "legacy"
    assert _value(groovy_results, f"forRow|config-defaults|{species}") == {}


@needs_nextflow
@pytest.mark.parametrize("species", SPECIES)
def test_groovy_for_species_runs_to_species_defaults(
    groovy_results: dict[str, dict[str, Any]], species: str
) -> None:
    """The executed ``forSpecies`` equals the pydantic species defaults."""
    assert _value(groovy_results, f"forSpecies|{species}") == species_defaults(species)  # type: ignore[arg-type]


@needs_nextflow
def test_groovy_for_row_in_map_first(groovy_results: dict[str, dict[str, Any]]) -> None:
    """map_first rows get every settings key, with row columns overriding."""
    keys = _groovy_strings(_groovy_constant(SETTINGS_SOURCE, "KEYS"))
    human_row = _value(groovy_results, "forRow|human|row-region")
    mouse_row = _value(groovy_results, "forRow|mouse|row-regions")

    assert list(human_row) == keys
    assert human_row == {
        "clustering_squidpy_mode": "map_first",
        "clustering_squidpy_table_key_suffix": "mapfirst",
        "annotation_anatomical_region": "frontal_cortex",
        "annotation_mouse_section_regions": None,
        "annotation_mode_mapmycells_stage": "legacy",
        "mender_unassigned_state_policy": "exclude_from_features",
    }
    assert (
        _value(groovy_results, "forRow|human|param-region")[
            "annotation_anatomical_region"
        ]
        == "frontal_cortex"
    )
    assert mouse_row["annotation_anatomical_region"] is None
    assert mouse_row["annotation_mouse_section_regions"] == "Isocortex;HPF"
    assert mouse_row["clustering_squidpy_table_key_suffix"] == "trial"
    assert (
        _value(groovy_results, "forRow|mouse|param-regions")[
            "annotation_mouse_section_regions"
        ]
        == "auto"
    )


@needs_nextflow
def test_groovy_settings_accessors(groovy_results: dict[str, dict[str, Any]]) -> None:
    """Rows without annotation keys read as legacy with no suffix."""
    assert _value(groovy_results, "accessors|legacy") == {
        "mode": "legacy",
        "legacy": True,
        "map_first": False,
        "suffix": "",
    }
    assert _value(groovy_results, "accessors|map_first") == {
        "mode": "map_first",
        "legacy": False,
        "map_first": True,
        "suffix": "mapfirst",
    }


@needs_nextflow
def test_groovy_run_mapmycells_rule(groovy_results: dict[str, dict[str, Any]]) -> None:
    """Legacy keeps the stage selection; map_first runs MMC only when asked."""
    expected = {
        "legacy|selected": True,
        "legacy|unselected": False,
        "map_first|stop-mapmycells": True,
        "map_first|stop-mender": False,
        "map_first|unselected": False,
        "map_first|skip": False,
    }
    for label, value in expected.items():
        assert _value(groovy_results, f"runMapMyCells|{label}") is value, label


@needs_nextflow
def test_groovy_completion_summary(groovy_results: dict[str, dict[str, Any]]) -> None:
    """Legacy runs print nothing; map_first runs list failed annotations."""
    assert _value(groovy_results, "completionSummary|legacy") == ""
    assert _value(groovy_results, "completionSummary|bad-species") == ""
    summary = _value(groovy_results, "completionSummary|mouse-map_first")
    assert summary.startswith(
        "Annotation summary (mouse, clustering_squidpy_mode map_first):"
    )
    assert "failed annotations: M1:reseg" in summary
    assert "refused panels: none" in summary


@needs_nextflow
def test_groovy_preflight(groovy_results: dict[str, dict[str, Any]]) -> None:
    """Preflight ignores legacy rows and refuses map_first until M5."""
    assert _value(groovy_results, "preflight|legacy") == []
    clustering = _value(groovy_results, "preflight|map_first|clustering")
    assert len(clustering) == 1
    assert "map_first is not available yet for P1" in clustering[0]
    assert _value(groovy_results, "preflight|map_first|no-clustered-stage") == []
    # A human row that would build WHB / SEA-AD bundles needs the held-out
    # donor's sources while resolvability is on (the default).
    human = "\n".join(
        _value(groovy_results, "preflight|map_first|human-no-selfmap-sources")
    )
    assert "whb_frontal_supc_clus, seaad_mr_panel need" in human
    assert "annotation_whb_region_precompute_source" in human
    # A mouse row that would build wmb_panel needs the self-map test cells
    # while resolvability is on (the default).
    mouse = "\n".join(_value(groovy_results, "preflight|map_first|mouse-no-test-cells"))
    assert "map_first is not available yet for M1" in mouse
    assert "wmb_panel needs annotation_wmb_selfmap_test_cells_path" in mouse
    invalid = "\n".join(_value(groovy_results, "preflight|map_first|invalid"))
    for expected in (
        "Unknown annotation_panel_mode 'bogus'",
        "Unknown human references in annotation_human_references",
        "annotation_ctm_version must name",
        "anatomical_region 'temporal_cortex'",
        "Missing annotation_gene_id_overrides_csv",
    ):
        assert expected in invalid
    assert "not available yet" not in invalid
    empty_suffix = _value(groovy_results, "preflight|map_first|empty-suffix")
    assert len(empty_suffix) == 1
    assert "overwrite the legacy one" in empty_suffix[0]
    assert "OD-A3" in empty_suffix[0]
    regions = _value(groovy_results, "preflight|map_first|mouse-regions")
    assert len(regions) == 1
    assert "Unknown mouse_section_regions 'Isocortex;Cortex'" in regions[0]
