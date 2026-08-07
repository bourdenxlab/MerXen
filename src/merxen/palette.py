"""The MerXen brand palette.

This module is the single source of truth for MerXen's colours. Everything that
draws something branded — figures, the pipeline metro map, README badges —
should take its colours from here rather than hardcoding hex strings.

The palette derives from the title logo in ``assets/MerXen_title.svg``:
``MERSCOPE`` is the purple of "Mer", ``XENIUM`` the teal of "Xen", and
``PAIRED`` the blue midpoint of the underline gradient, where the two halves
meet. Anything that is neither platform nor a paired result — the toggleable
stages on the metro map, an accent on a plot — uses ``OPTIONAL``.

``tests/test_workflows/test_metro_map.py`` checks the metro map source in
``assets/metro_map.mmd`` against these values, so the diagram cannot silently
drift away from the rest of the project.

Note that this palette is chosen for brand consistency, not for colour-vision
accessibility: ``MERSCOPE`` and ``XENIUM`` are not reliably separable under
simulated tritanopia or deuteranopia. Where a figure must be read by colour
alone, pair these with a redundant non-colour cue — a dash pattern, a marker
shape, or a direct label — as the metro map does for its toggleable line.
"""

from __future__ import annotations

from typing import Final

#: Vizgen MERSCOPE. The purple of "Mer" in the title logo.
MERSCOPE: Final = "#8350FF"

#: 10x Genomics Xenium. The teal of "Xen" in the title logo.
XENIUM: Final = "#17BCAC"

#: Paired-section results, where the two platforms converge. The blue midpoint
#: of the title logo's underline gradient.
PAIRED: Final = "#3980BE"

#: Accent for anything that is neither platform nor a paired result: optional
#: stages, highlights, callouts. Deliberately outside the purple/teal range so
#: it never reads as a platform.
OPTIONAL: Final = "#EE7733"

#: Light neutral used behind grouped content, e.g. metro map section boxes.
SURFACE: Final = "#EDEDED"

#: The two platform colours, keyed by the platform names used throughout the
#: pipeline (``merscope`` / ``xenium``), for indexing straight off a config or
#: a dataframe column.
PLATFORM_COLOURS: Final[dict[str, str]] = {
    "merscope": MERSCOPE,
    "xenium": XENIUM,
}

#: Every named brand colour, keyed by role. Used by the metro map drift test.
BRAND_COLOURS: Final[dict[str, str]] = {
    "merscope": MERSCOPE,
    "xenium": XENIUM,
    "paired": PAIRED,
    "optional": OPTIONAL,
}
