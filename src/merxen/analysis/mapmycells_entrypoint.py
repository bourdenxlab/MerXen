"""MerXen wrapper around the local MapMyCells CLI."""

from __future__ import annotations

from merxen.analysis.mapmycells_gpu_compat import apply_mapmycells_gpu_compat_patch


def main() -> None:
    """Run MapMyCells with MerXen's GPU compatibility patch applied."""
    apply_mapmycells_gpu_compat_patch()

    from cell_type_mapper.cli.from_specified_markers import main as mapmycells_main

    mapmycells_main()


if __name__ == "__main__":
    main()
