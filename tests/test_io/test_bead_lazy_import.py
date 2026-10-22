"""The bead-removal package is lazy: default runs do not load the correction code."""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

import merxen.io.bead_removal as bead_removal

# The submodules the image-preprocessing record needs; nothing else may load.
_LIGHT_SUBMODULES = {"params", "provenance"}


def test_default_entry_points_import_only_the_light_submodules() -> None:
    code = (
        "import sys\n"
        "import merxen.io.builders.merscope\n"
        "import merxen.segmentation.pipeline\n"
        "import merxen.enrichment.enrich\n"
        "import merxen.mask_image_quantification\n"
        "prefix = 'merxen.io.bead_removal.'\n"
        "names = [m for m in sys.modules if m.startswith(prefix)]\n"
        "loaded = sorted(m[len(prefix) :] for m in names)\n"
        "print(','.join(loaded))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )

    assert done.returncode == 0, done.stderr[-2000:]
    assert set(done.stdout.strip().split(",")) == _LIGHT_SUBMODULES


def test_every_public_name_resolves_to_its_submodule() -> None:
    assert sorted(bead_removal.__all__) == sorted(bead_removal._EXPORTS)
    for name, submodule in bead_removal._EXPORTS.items():
        module = importlib.import_module(f"merxen.io.bead_removal.{submodule}")
        assert getattr(bead_removal, name) is getattr(module, name)
        assert name in dir(bead_removal)


def test_submodules_are_reachable_as_attributes() -> None:
    assert bead_removal.driver is importlib.import_module(
        "merxen.io.bead_removal.driver"
    )
    assert bead_removal.backend is importlib.import_module(
        "merxen.io.bead_removal.backend"
    )


def test_an_unknown_name_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="no attribute 'nothing'"):
        bead_removal.nothing  # noqa: B018
