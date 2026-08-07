"""Guards for the pipeline overview metro map.

The map in ``assets/metro_map.mmd`` is hand-maintained, so its failure mode is
silent staleness: a stage is added to the workflow and the picture quietly stops
describing the pipeline. These tests fail the build instead.

They deliberately do not import ``nf_metro`` — the source is parsed with plain
regexes so the checks run for contributors who have not installed the dev extra.
"""

from __future__ import annotations

import itertools
import re
from pathlib import Path

import pytest

from merxen.palette import BRAND_COLOURS

REPO_ROOT = Path(__file__).resolve().parents[2]
MMD_PATH = REPO_ROOT / "assets" / "metro_map.mmd"
MODULES_DIR = REPO_ROOT / "workflows" / "modules"

# Processes intentionally absent from the map, with the reason. The map is a
# reader's overview, not a DAG dump; these carry no scientific meaning.
OMITTED_PROCESSES = {
    "ENSURE_PROSEG": "bootstrap step that installs the ProSeg binary",
    "VIEWER_CACHE": "derived cache for the interactive viewer",
    "VALIDATE_ANALYSIS_LAYER": "pre-QC guard with no artifacts of its own",
}

# Stations that stand for more than one process, or for none at all.
STATION_ALIASES = {
    "clustering_squidpy": {
        "CLUSTERING_SQUIDPY_PREPARE",
        "CLUSTERING_SQUIDPY_COMPUTE",
        "CLUSTERING_SQUIDPY_FINALIZE",
    },
    "mender": {
        "MENDER_PREPARE",
        "MENDER_COMPUTE",
        "MENDER_FINALIZE",
        "MENDER_IMPORT",
    },
}

# Decorative nodes: terminus icons and the segmentation branch that is a
# parameter of PROSEG_SEGMENT rather than a process of its own.
NON_PROCESS_STATIONS = {
    "merscope_in",
    "xenium_in",
    "mecr_out",
    "compare_out",
    "mapmycells_out",
    "proseg_hybrid",
}


@pytest.fixture(scope="module")
def mmd_text() -> str:
    """Return the metro map source."""
    return MMD_PATH.read_text()


def _nextflow_processes() -> set[str]:
    """Return every process name declared under ``workflows/modules/``."""
    pattern = re.compile(r"^process\s+([A-Z0-9_]+)\s*\{", re.MULTILINE)
    names: set[str] = set()
    for module in MODULES_DIR.glob("*.nf"):
        names.update(pattern.findall(module.read_text()))
    return names


def _stations(mmd_text: str) -> set[str]:
    """Return every station id declared in the map source.

    Station declarations are ``id[Label]`` lines inside a subgraph; edge lines
    (``a -->|line| b``) reference ids that are always declared elsewhere.
    """
    pattern = re.compile(r"^\s*([a-z0-9_]+)\[", re.MULTILINE)
    return set(pattern.findall(mmd_text))


def _declared_lines(mmd_text: str) -> dict[str, str]:
    """Return ``{line_id: hex_colour}`` for every ``%%metro line:`` directive."""
    pattern = re.compile(
        r"^%%metro line:\s*([a-z_]+)\s*\|[^|]*\|\s*(#[0-9A-Fa-f]{6})", re.MULTILINE
    )
    return dict(pattern.findall(mmd_text))


def test_every_pipeline_process_has_a_station(mmd_text: str) -> None:
    """Adding a Nextflow process must be reflected in the metro map."""
    aliased = set(itertools.chain.from_iterable(STATION_ALIASES.values()))
    stations = _stations(mmd_text)
    expected = {
        name
        for name in _nextflow_processes()
        if name not in OMITTED_PROCESSES and name not in aliased
    }
    missing = sorted(name for name in expected if name.lower() not in stations)
    assert not missing, (
        f"Nextflow processes with no station in {MMD_PATH.name}: {missing}. "
        "Add a station, or record it in OMITTED_PROCESSES with a reason."
    )


def test_aliased_processes_still_exist(mmd_text: str) -> None:
    """A collapsed station must keep covering processes that really exist."""
    processes = _nextflow_processes()
    for station, covered in STATION_ALIASES.items():
        assert station in _stations(mmd_text), f"alias station {station!r} is gone"
        stale = sorted(covered - processes)
        assert not stale, f"{station!r} claims to cover removed processes: {stale}"


def test_no_station_refers_to_a_missing_process(mmd_text: str) -> None:
    """Renaming or deleting a process must not leave an orphan station."""
    processes = {name.lower() for name in _nextflow_processes()}
    orphans = sorted(
        station
        for station in _stations(mmd_text)
        if station not in NON_PROCESS_STATIONS
        and station not in STATION_ALIASES
        and station not in processes
    )
    assert not orphans, (
        f"Stations naming no Nextflow process: {orphans}. "
        "Rename the station, or add it to NON_PROCESS_STATIONS if decorative."
    )


def test_omitted_processes_are_still_real(mmd_text: str) -> None:
    """The omission list must not accumulate names that no longer exist."""
    stale = sorted(set(OMITTED_PROCESSES) - _nextflow_processes())
    assert not stale, f"OMITTED_PROCESSES lists removed processes: {stale}"


# --- brand palette ----------------------------------------------------------


def test_line_colours_match_the_brand_palette(mmd_text: str) -> None:
    """The map's lines must be the brand colours from ``merxen.palette``.

    The map is the most visible thing in the repo, so it is the place a stale
    colour is most likely to be noticed and least likely to be chased down.
    Keying it to the palette module means a rebrand is one edit, not a hunt.
    """
    declared = _declared_lines(mmd_text)
    assert {k.lower(): v.upper() for k, v in declared.items()} == {
        k: v.upper() for k, v in BRAND_COLOURS.items()
    }, (
        f"metro line colours {declared} do not match merxen.palette "
        f"{BRAND_COLOURS}. Update whichever is stale, then re-run "
        "scripts/render_metro_map.py."
    )


def test_no_two_lines_share_a_colour(mmd_text: str) -> None:
    """Two lines drawn in the same colour make the map unreadable."""
    lines = _declared_lines(mmd_text)
    assert len(lines) >= 2, "expected at least two %%metro line: directives"
    for first, second in itertools.combinations(sorted(lines), 2):
        assert (
            lines[first].upper() != lines[second].upper()
        ), f"lines {first!r} and {second!r} are both {lines[first]}"


def test_rendered_svgs_are_current(mmd_text: str) -> None:
    """The committed SVGs must exist and mention every station label."""
    for name in RENDERED_SVGS:
        svg = REPO_ROOT / "docs" / "images" / name
        assert svg.is_file(), f"{name} is missing; run scripts/render_metro_map.py"
        text = svg.read_text()
        for line_id, colour in _declared_lines(mmd_text).items():
            assert colour.lower() in text.lower(), (
                f"{name} does not use the {line_id!r} colour {colour}; "
                "re-run scripts/render_metro_map.py"
            )


# The map is embedded at the width of a README or docs column, so a very wide,
# very short figure renders its labels too small to read. Keep it compact.
MAX_RENDERED_WIDTH = 2000
MAX_ASPECT_RATIO = 3.0
RENDERED_SVGS = ("merxen_metro_map.svg", "merxen_metro_map_animated.svg")


def _station_labels(mmd_text: str) -> set[str]:
    """Return every non-empty station label declared in the map source."""
    pattern = re.compile(r"^\s*[a-z0-9_]+\[([^\]]+)\]", re.MULTILINE)
    return {m.strip() for m in pattern.findall(mmd_text) if m.strip()}


@pytest.mark.parametrize("name", RENDERED_SVGS)
def test_rendered_map_stays_compact(name: str) -> None:
    """Guard the aspect ratio that keeps the labels legible when embedded."""
    svg = (REPO_ROOT / "docs" / "images" / name).read_text()
    match = re.search(r'width="(\d+)" height="(\d+)"', svg)
    assert match is not None, f"{name} has no intrinsic width/height"
    width, height = int(match.group(1)), int(match.group(2))
    assert width <= MAX_RENDERED_WIDTH, (
        f"{name} is {width}px wide (limit {MAX_RENDERED_WIDTH}); it will render "
        "too small when embedded. Re-check the layout rather than raising this."
    )
    assert width / height <= MAX_ASPECT_RATIO, (
        f"{name} has aspect ratio {width / height:.1f}:1 "
        f"(limit {MAX_ASPECT_RATIO}:1); the figure has stretched back into a "
        "wide strip."
    )


@pytest.mark.parametrize("name", RENDERED_SVGS)
def test_station_labels_are_not_wrapped(mmd_text: str, name: str) -> None:
    """No station label may hyphenate across lines in the rendered SVG.

    nf-metro splits an over-long label into several ``<text>`` elements, so a
    wrapped label never appears as one element's full contents. This is the
    failure that shows up when --font-scale is raised without re-checking the
    render.
    """
    svg = (REPO_ROOT / "docs" / "images" / name).read_text()
    rendered = {
        re.sub(r"<[^>]+>", "", element).strip()
        for element in re.findall(r"<text[^>]*>.*?</text>", svg, re.DOTALL)
    }
    wrapped = sorted(
        label for label in _station_labels(mmd_text) if label not in rendered
    )
    assert not wrapped, (
        f"{name} wraps these station labels mid-word: {wrapped}. Shorten the "
        "label or lower --font-scale in scripts/render_metro_map.py."
    )
