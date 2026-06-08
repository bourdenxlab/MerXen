"""Runtime checks for optional alignment dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version


@dataclass(frozen=True)
class AlignmentDependencyStatus:
    """Result of checking the optional Spateo alignment stack."""

    ok: bool
    message: str
    versions: dict[str, str] = field(default_factory=dict)


def check_alignment_dependencies() -> AlignmentDependencyStatus:
    """Return whether the shimmed Spateo alignment imports are available."""
    versions = {
        package: _package_version(package)
        for package in (
            "spateo-release",
            "dynamo-release",
            "anndata",
            "cellpose",
        )
    }
    try:
        from merxen.alignment.register import _apply_spateo_import_shims

        _apply_spateo_import_shims()
        import spateo  # noqa: F401
        from spateo.alignment.morpho_alignment import Morpho_pairwise  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return AlignmentDependencyStatus(
            ok=False,
            message=(
                "Spateo alignment dependencies are not importable after "
                f"compatibility shims: {type(exc).__name__}: {exc}"
            ),
            versions=versions,
        )

    return AlignmentDependencyStatus(
        ok=True,
        message="Spateo alignment dependencies are importable.",
        versions=versions,
    )


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not installed"
