"""Load the CI tooling of ``scripts/ci`` as modules."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

CI_DIR = Path(__file__).resolve().parents[2] / "scripts" / "ci"


def load_ci_script(name: str) -> ModuleType:
    """Import ``scripts/ci/<name>.py`` (cached in ``sys.modules``)."""
    key = f"_ci_script_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, CI_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve string annotations through sys.modules.
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module
