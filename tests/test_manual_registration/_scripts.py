"""Load the one-off scripts of ``scripts/manual_registration`` as modules."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts" / "manual_registration"


def load_script(name: str) -> ModuleType:
    """Import ``scripts/manual_registration/<name>.py`` (cached in ``sys.modules``)."""
    key = f"_manual_registration_script_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, SCRIPTS_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve string annotations through sys.modules.
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module
