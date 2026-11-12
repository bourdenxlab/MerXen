"""The clustering code must import without cell_type_mapper or SpatialData.

Legacy ``CLUSTERING_SQUIDPY_COMPUTE`` runs ``merxen.clustering_squidpy_stages``
in the GPU env, which has neither package (plan §3.5, R-eng M6). The
subprocesses below block them through ``sys.meta_path`` and import the stage
entry point, every ``merxen.clustering`` module and the annotation contract
modules; ``merxen.clustering`` and ``merxen.table_keys`` must also import with
nothing heavier than numpy, pandas and pydantic.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

GPU_ENV_BLOCKED = ("cell_type_mapper", "spatialdata", "spatialdata_io")
GPU_ENV_MODULES = (
    "merxen.clustering_squidpy_stages",
    "merxen.annotation.schema",
    "merxen.annotation.vocab",
    "merxen.annotation.config",
    "merxen.table_keys",
)
LIGHT_BLOCKED = (
    "anndata",
    "cell_type_mapper",
    "dask",
    "h5py",
    "matplotlib",
    "scanpy",
    "scipy",
    "sklearn",
    "spatialdata",
    "spatialdata_io",
    "squidpy",
    "torch",
    "zarr",
)
LIGHT_MODULES = ("merxen.table_keys",)

SCRIPT = textwrap.dedent(
    """
    import importlib
    import pkgutil
    import sys

    BLOCKED = set({blocked!r})

    class Blocker:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in BLOCKED:
                raise ImportError(f"blocked import: {{name}}")
            return None

    sys.meta_path.insert(0, Blocker())
    import merxen.clustering

    modules = list({modules!r})
    modules += sorted(
        f"merxen.clustering.{{info.name}}"
        for info in pkgutil.iter_modules(merxen.clustering.__path__)
    )
    assert "merxen.clustering.cellset" in modules, modules
    for module in modules:
        importlib.import_module(module)
    loaded = sorted(name for name in sys.modules if name.split(".")[0] in BLOCKED)
    assert not loaded, loaded
    print("ok")
    """
)


def _run_isolated(blocked: tuple[str, ...], modules: tuple[str, ...]) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "src"), env.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT.format(blocked=blocked, modules=modules)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok"), result.stdout


def test_clustering_stages_import_without_ctm_or_spatialdata() -> None:
    """The GPU-env entry point and the contract modules import there."""
    _run_isolated(GPU_ENV_BLOCKED, GPU_ENV_MODULES)


def test_clustering_package_imports_with_numpy_pandas_and_pydantic_only() -> None:
    """``merxen.clustering`` and ``merxen.table_keys`` need no heavy package."""
    _run_isolated(LIGHT_BLOCKED, LIGHT_MODULES)
