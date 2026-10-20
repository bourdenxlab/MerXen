#!/usr/bin/env python
"""Render the pipeline overview metro map from ``assets/metro_map.mmd``.

Produces two committed SVGs under ``docs/images/``:

- ``merxen_metro_map.svg`` — static, embedded in the docs
- ``merxen_metro_map_animated.svg`` — animated, embedded in the README

Both are rendered with the light theme and keep nf-metro's
``prefers-color-scheme`` block, so one file adapts to the reader's GitHub theme.
Run this after adding or renaming a pipeline stage in the ``.mmd`` source.

nf-metro exposes no CLI flag for stroke weight or legend sizing, so this script
patches the light theme in place and then delegates to nf-metro's own CLI. That
reaches into nf-metro internals, which upstream documents as not
semver-stable — hence the ``nf-metro>=1.1,<1.2`` pin in ``pyproject.toml``. If a
future version moves these names the script fails loudly at import rather than
silently rendering unstyled output.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "assets" / "metro_map.mmd"
OUTPUT_DIR = REPO_ROOT / "docs" / "images"
STATIC_SVG = OUTPUT_DIR / "merxen_metro_map.svg"
ANIMATED_SVG = OUTPUT_DIR / "merxen_metro_map_animated.svg"

# Light-theme fields overridden before rendering. The defaults are tuned for a
# map viewed at full size; ours is embedded at the width of a README column, so
# the strokes need more weight to survive being scaled down.
THEME_OVERRIDES = {
    "line_width": 5.5,  # default 3.0
    "legend_font_size": 19.0,  # default 16.0, before --font-scale is applied
}

# Leave ``station_radius`` at its 6.0 default. Enlarging it to match the heavier
# strokes widens each station enough to hyphenate "Spatial genes" and
# "Clustering" at the current spacing, for no real visual gain.

# Length of the colour swatch in each legend row (default 24.0). The swatch
# takes its thickness from ``line_width`` above; this only sets how far it runs,
# and the legend box reflows around it.
LEGEND_SWATCH_WIDTH = 56.0

# Vertical pitch between legend rows (default 24.0). This one is *not* scaled by
# --font-scale, so at our 26.6px rendered legend type the stock 24.0 is tighter
# than the text is tall. The legend sits in empty space below the map, so the
# extra height costs nothing on the canvas.
LEGEND_LINE_HEIGHT = 40.0

# nf-metro's dashed style is a fixed "8,4", which assumes the default 3.0
# stroke. Round line caps extend half the stroke width past each dash end, so at
# 5.5 the caps close the 4px gap completely and the "toggleable" line renders
# solid — losing the redundant, colour-independent cue the palette's
# accessibility argument leans on. Scale the pattern with the stroke instead.
DASH_ON = 12.0
DASH_OFF = 10.0

# Grid column whose sections should share a left edge. nf-metro right-aligns
# every section in a column that holds an RL or TB section, so column 0 —
# "Ingest and build" on the top row and the wider "Annotation and DE" on the
# return row — ends up flush right, and the wider box juts out to the left with
# the bottom-left legend trailing after it. Aligning the left edges instead
# reads as a deliberate margin. See _align_column_left_edges.
LEFT_ALIGNED_COLUMN = 0

# Shared render options.
#
# --x-spacing 90 and --font-scale 1.4 were chosen together by sweeping both and
# checking every station label still renders as one unwrapped line. They are not
# independent: raising the font past 1.4 at this spacing hyphenates
# "Cellpose-SAM cells", and *widening* the spacing makes the on-screen text
# smaller, because the figure grows faster than the type does. Re-run the sweep
# rather than nudging either value alone.
#
# --embed-font inlines Inter so the map renders the same on any host.
# --validate --strict turns a geometry defect into a failed build rather than a
# subtly broken picture.
COMMON_OPTIONS = [
    "--theme",
    "light",
    "--x-spacing",
    "90",
    "--font-scale",
    "1.4",
    "--embed-font",
    "--validate",
    "--strict",
]


def apply_style_overrides() -> None:
    """Patch nf-metro's light theme and legend geometry in place."""
    from nf_metro.render import constants, legend
    from nf_metro.themes.light import LIGHT_THEME

    for field, value in THEME_OVERRIDES.items():
        if not hasattr(LIGHT_THEME, field):
            raise AttributeError(
                f"nf-metro's light theme has no {field!r} field; the styling "
                "overrides in this script need updating for this version."
            )
        setattr(LIGHT_THEME, field, value)

    for name, value in (
        ("LEGEND_SWATCH_WIDTH", LEGEND_SWATCH_WIDTH),
        ("LEGEND_LINE_HEIGHT", LEGEND_LINE_HEIGHT),
    ):
        if not hasattr(legend, name):
            raise AttributeError(
                f"nf_metro.render.legend no longer defines {name}; the legend "
                "overrides in this script need updating for this version."
            )
        setattr(legend, name, value)

    if "dashed" not in constants.STROKE_DASHARRAY:
        raise KeyError(
            "nf_metro.render.constants.STROKE_DASHARRAY has no 'dashed' entry; "
            "the dash override in this script needs updating for this version."
        )
    # Mutated in place: line_style_kwargs() reads this dict at call time.
    constants.STROKE_DASHARRAY["dashed"] = f"{DASH_ON:g},{DASH_OFF:g}"


def align_column_left_edges(column: int = LEFT_ALIGNED_COLUMN) -> None:
    """Make every section in ``column`` share the rightmost of their left edges.

    nf-metro has no alignment directive: :func:`_compute_section_offsets`
    right-aligns a column as soon as any section in it runs ``RL`` or ``TB``,
    which is unavoidable here because the return row flows right to left. The
    two sections in column 0 are different widths, so a shared right edge means
    the wider one overhangs to the left.

    ``_pack_cells`` is the last thing ``_compute_section_offsets`` does, and by
    then every section's ``offset_x`` is final, so wrapping it gives a clean
    hook to nudge the overhanging section back inwards. Moving the wider box
    right — rather than pulling the narrower one left — spends slack that
    already exists on the return row instead of opening a new gap on the top
    row.
    """
    from nf_metro.layout import section_placement

    if not hasattr(section_placement, "_pack_cells"):
        raise AttributeError(
            "nf_metro.layout.section_placement no longer defines _pack_cells; "
            "the column-alignment patch in this script needs updating."
        )

    original = section_placement._pack_cells

    def patched(
        scoped: dict[str, Any],
        packs: dict[tuple[int, int], list[str]],
        col_offsets: dict[int, float],
        col_widths: dict[int, float],
        right_align_cols: set[int],
        gap: float,
    ) -> None:
        original(scoped, packs, col_offsets, col_widths, right_align_cols, gap)
        members = [
            section
            for section in scoped.values()
            if section.grid_col == column and section.grid_col_span == 1
        ]
        if len(members) < 2:
            return
        # Align the *drawn* left edges, not the offsets. A section's box starts
        # at offset_x + bbox_x, and bbox_x differs between these two: the
        # terminus icons on "Ingest and build" push its bbox left to make room
        # for them. Aligning the raw offsets leaves the boxes 8px apart.
        left_edge = max(section.offset_x + section.bbox_x for section in members)
        for section in members:
            section.offset_x = left_edge - section.bbox_x

    section_placement._pack_cells = patched


def render(destination: Path, *extra: str) -> None:
    """Render the map to ``destination`` with the shared options plus ``extra``."""
    from nf_metro.cli import cli

    argv = ["render", str(SOURCE), "-o", str(destination), *COMMON_OPTIONS, *extra]
    print(f"Rendering {destination.relative_to(REPO_ROOT)}")
    cli.main(args=argv, standalone_mode=False)


def ensure_trailing_newline(path: Path) -> None:
    """Append the trailing newline pre-commit's end-of-file-fixer requires."""
    content = path.read_bytes()
    if content and not content.endswith(b"\n"):
        path.write_bytes(content + b"\n")


def main() -> int:
    """Validate the source, render both SVGs, and report their sizes."""
    try:
        from nf_metro.cli import cli
    except ModuleNotFoundError:
        print(
            "nf-metro not found. Install the dev extra: uv pip install -e '.[dev]'",
            file=sys.stderr,
        )
        return 1

    print(f"Validating {SOURCE.relative_to(REPO_ROOT)}")
    cli.main(args=["validate", str(SOURCE)], standalone_mode=False)

    apply_style_overrides()
    align_column_left_edges()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    render(STATIC_SVG)
    render(ANIMATED_SVG, "--animate")

    for path in (STATIC_SVG, ANIMATED_SVG):
        ensure_trailing_newline(path)

    print("Done:")
    subprocess.run(["du", "-h", str(STATIC_SVG), str(ANIMATED_SVG)], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
