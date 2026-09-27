"""Smoke tests for the M3 stage-C2 shadow scripts (``scripts/acceptance/``)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from merxen.annotation.shadow import (
    E1_REFEREE_MARKERS,
    evaluate_human_rules,
    rule_inputs_from_provisional,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts" / "acceptance"
NAMES = (
    "shadow_baselines",
    "shadow_ll",
    "heldout_genes",
    "shadow_x1",
    "shadow_flags",
    "shadow_e8",
    "shadow_glial_jsd",
)


def _load(name: str) -> ModuleType:
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scripts() -> dict[str, ModuleType]:
    return {name: _load(name) for name in NAMES}


@pytest.mark.parametrize("name", NAMES[1:])
def test_script_help_runs(scripts: dict[str, ModuleType], name: str) -> None:
    with pytest.raises(SystemExit) as raised:
        scripts[name].main(["--help"])
    assert raised.value.code == 0


def _sample(baselines: ModuleType, n_cells: int = 40) -> object:
    rng = np.random.default_rng(0)
    astro = np.arange(n_cells) % 2 == 0
    labels = pd.DataFrame(
        {
            "total_counts": np.where(np.arange(n_cells) < 10, 30, 100),
            "ct_lineage_name": np.where(astro, "Astrocytes", "Neurons"),
            "ct_lineage_raw": np.full(n_cells, 0.9),
            "ct_broad_name": np.where(astro, "Astrocytes", "Neurons"),
            "ct_broad_raw": np.full(n_cells, 0.9),
            "mmc_whb_supercluster_name": np.where(
                astro, "Astrocyte", "Upper-layer intratelencephalic"
            ),
            "mmc_whb_supercluster_bp": np.full(n_cells, 0.9),
        },
        index=pd.Index([f"c{i}" for i in range(n_cells)], name="cell_id"),
    )
    sea_broad = labels["ct_broad_name"].to_numpy(object).copy()
    sea_broad[:4] = "Microglia"  # SEA-AD disagrees on four cells below 60 counts
    sea = pd.DataFrame(
        {"broad": sea_broad, "broad_raw": np.full(n_cells, 0.9)}, index=labels.index
    )
    scores = np.zeros((n_cells, len(E1_REFEREE_MARKERS)))
    classes = list(E1_REFEREE_MARKERS)
    scores[astro, classes.index("Astrocytes")] = 0.3
    scores[~astro, classes.index("Neurons")] = 0.3
    rules = {
        vote: evaluate_human_rules(
            rule_inputs_from_provisional(labels, sea),
            platform="MERSCOPE",
            second_vote=vote,
        )
        for vote in baselines.VOTES
    }
    return baselines.Sample(
        sample_id="PX_MERSCOPE",
        platform="MERSCOPE",
        labels=labels,
        sea=sea,
        xy=rng.random((n_cells, 2)) * 2000,
        legacy_broad=np.full(n_cells, "Neurons", dtype=object),
        scores=scores,
        aligned_frame=True,
        rules=rules,
    )


def test_ll_score_sample_and_decision(scripts: dict[str, ModuleType]) -> None:
    sample = _sample(scripts["shadow_baselines"])
    labels = sample.labels  # type: ignore[attr-defined]
    calls = pd.DataFrame(
        {
            "ll_broad_name": labels["ct_broad_name"].to_numpy(object),
            "ll_mix_fraction": np.zeros(len(labels)),
        },
        index=labels.index,
    )
    row, referee = scripts["shadow_ll"].score_sample(
        sample,
        calls,
        pair="P7513",
        seg="proseg_hybrid",
        variant="capped",
        n_segmented=None,
    )
    # LL agrees with WHB on the four cells SEA-AD disputes below 60 counts.
    assert row["od_b8_coverage_gain"] == pytest.approx(4 / 40)
    assert row["n_gained_v11"] == 4
    assert row["cost_sea_below60"] == pytest.approx(4 / 40)
    assert row["ll_whb_agree_all"] == 1.0
    assert referee[0]["comparison"].startswith("WHB argmax vs LL")
    metrics = pd.DataFrame([row, {**row, "pair": "P1212"}])
    decisions = {
        item["decision"]: item for item in scripts["shadow_ll"].decision_rows(metrics)
    }
    assert decisions["OD-B8"]["passes"]  # a 10-point coverage gain
    assert decisions["OD-B13"]["max_cost_sea_below60"] == pytest.approx(0.1)


def test_heldout_h4_rows_count_passing_classes(scripts: dict[str, ModuleType]) -> None:
    enrichment = pd.DataFrame(
        {
            "pair": ["P7513"] * 7,
            "platform": ["XENIUM"] * 7,
            "variant": ["set_a"] * 7,
            "label_set": ["heldout_argmax"] * 7,
            "broad_class": list("ABCDEFG"),
            "passes": [True] * 6 + [False],
        }
    )
    [row] = scripts["heldout_genes"].h4_rows(enrichment)
    assert row["n_pass"] == 6 and row["h4_pass"] and row["failing"] == "G"


def test_x1_labelling_matrices_have_the_three_kinds(
    scripts: dict[str, ModuleType],
) -> None:
    sample = _sample(scripts["shadow_baselines"])
    matrices, argmax, coverage = scripts["shadow_x1"].labelling_matrices(
        sample.labels,  # type: ignore[attr-defined]
        sample,
        None,
    )
    assert set(matrices) == {"soft", "argmax", "confident"}
    assert len(argmax) == 40
    assert coverage == pytest.approx(36 / 40)
