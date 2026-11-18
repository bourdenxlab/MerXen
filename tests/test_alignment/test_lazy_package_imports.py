"""The manual registration must not import the alignment pipeline or Cellpose."""

from __future__ import annotations

import subprocess
import sys
import textwrap

HEAVY_MODULES = ("torch", "cellpose", "valis", "kornia")


def _run(code: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    return result.stdout.strip()


def test_importing_manual_registration_leaves_the_heavy_stacks_unloaded() -> None:
    # A registration run measured 0.5 GB less resident memory and 3 s less import
    # time once enrichment (Cellpose, PyTorch) and the alignment pipeline (VALIS)
    # were no longer pulled in at import.
    loaded = _run(
        f"""
        import sys
        import merxen.manual_registration.register
        print(",".join(m for m in {HEAVY_MODULES!r} if m in sys.modules))
        """
    )

    assert loaded == ""


def test_alignment_package_still_exposes_its_public_names_lazily() -> None:
    names = _run(
        """
        import sys
        import merxen.alignment as alignment
        before = "merxen.alignment.pipeline" in sys.modules
        function = alignment.run_alignment_pipeline
        from merxen.alignment import frames
        print(before, callable(function), function.__module__,
              frames.__name__, sorted(alignment.__all__) == alignment.__all__)
        """
    )

    assert names == (
        "False True merxen.alignment.pipeline merxen.alignment.frames True"
    )


def test_alignment_package_raises_attribute_error_for_an_unknown_name() -> None:
    result = _run(
        """
        import merxen.alignment as alignment
        try:
            alignment.not_a_name
        except AttributeError as error:
            print("AttributeError", "not_a_name" in str(error))
        """
    )

    assert result == "AttributeError True"
