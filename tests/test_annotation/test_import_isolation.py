"""The annotation contract modules must import in the GPU clustering env.

That env has numpy, pandas and pydantic but neither ``cell_type_mapper`` nor
SpatialData (plan §3.5), so these modules may import nothing heavier.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULES = (
    "merxen.annotation.schema",
    "merxen.annotation.vocab",
    "merxen.annotation.config",
    "merxen.annotation.provenance",
    "merxen.annotation.samplesheet_columns",
    # M3c: the downgrade-only real-data QC (plan §8.8), used by RESOLVE.
    "merxen.annotation.real_qc",
)
BLOCKED = (
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
    "yaml",
    "zarr",
)

SCRIPT = textwrap.dedent(
    """
    import importlib
    import sys

    BLOCKED = set({blocked!r})

    class Blocker:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in BLOCKED:
                raise ImportError(f"blocked import: {{name}}")
            return None

    sys.meta_path.insert(0, Blocker())
    for module in {modules!r}:
        importlib.import_module(module)
    from merxen.annotation.config import AnnotationConfig
    from merxen.annotation.vocab import load_vocab

    AnnotationConfig(species="mouse")
    load_vocab("whb_supercluster")
    loaded = sorted(name for name in sys.modules if name.split(".")[0] in BLOCKED)
    assert not loaded, loaded
    print("ok")
    """
)


def test_contract_modules_import_without_heavy_dependencies() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "src"), env.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT.format(blocked=BLOCKED, modules=MODULES)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
