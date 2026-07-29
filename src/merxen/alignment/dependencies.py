"""Runtime checks for optional alignment dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version


@dataclass(frozen=True)
class AlignmentDependencyStatus:
    """Result of checking one optional alignment dependency stack."""

    ok: bool
    message: str
    versions: dict[str, str] = field(default_factory=dict)


def check_alignment_dependencies(
    backend: str = "valis",
) -> AlignmentDependencyStatus:
    """Return whether the requested alignment backend imports are available."""
    normalized = str(backend).strip().lower()
    if normalized == "legacy_spateo":
        return _check_legacy_spateo_dependencies()
    if normalized != "valis":
        return AlignmentDependencyStatus(
            ok=False,
            message=f"Unknown alignment backend: {backend!r}",
        )

    versions = {
        package: _package_version(package)
        for package in (
            "valis-wsi",
            "opencv-contrib-python-headless",
            "scikit-image",
            "torch",
            "torchvision",
            "kornia",
            "SimpleITK",
            "pyvips",
            "jpype1",
        )
    }
    try:
        from merxen.alignment.valis_compat import apply_valis_numpy_compatibility

        apply_valis_numpy_compatibility()
        import cv2
        from valis import (  # noqa: F401
            feature_detectors,
            feature_matcher,
            non_rigid_registrars,
            preprocessing,
            registration,
        )
        from valis.preprocessing import ImageProcesser  # noqa: F401

        if not hasattr(cv2, "optflow"):
            raise RuntimeError("OpenCV contrib optflow module is unavailable")
    except Exception as exc:  # noqa: BLE001
        return AlignmentDependencyStatus(
            ok=False,
            message=(
                "VALIS alignment dependencies are not importable: "
                f"{type(exc).__name__}: {exc}"
            ),
            versions=versions,
        )
    if versions["valis-wsi"] != "1.2.0":
        return AlignmentDependencyStatus(
            ok=False,
            message=(
                "VALIS alignment requires valis-wsi==1.2.0; "
                f"found {versions['valis-wsi']}"
            ),
            versions=versions,
        )
    return AlignmentDependencyStatus(
        ok=True,
        message="VALIS 1.2 alignment dependencies are importable.",
        versions=versions,
    )


def _check_legacy_spateo_dependencies() -> AlignmentDependencyStatus:
    """Return whether the shimmed legacy Spateo imports are available."""
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
        from merxen.alignment.legacy_spateo import _apply_spateo_import_shims

        _apply_spateo_import_shims()
        import spateo  # noqa: F401
        from spateo.alignment.morpho_alignment import Morpho_pairwise  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return AlignmentDependencyStatus(
            ok=False,
            message=(
                "Legacy Spateo alignment dependencies are not importable after "
                f"compatibility shims: {type(exc).__name__}: {exc}"
            ),
            versions=versions,
        )

    return AlignmentDependencyStatus(
        ok=True,
        message="Legacy Spateo alignment dependencies are importable.",
        versions=versions,
    )


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not installed"
