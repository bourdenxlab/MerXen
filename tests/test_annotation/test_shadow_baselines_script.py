"""Smoke tests for ``scripts/acceptance/shadow_baselines.py`` on synthetic samples."""

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
SCRIPT_PATH = REPO_ROOT / "scripts" / "acceptance" / "shadow_baselines.py"


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("shadow_baselines", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["shadow_baselines"] = module
    spec.loader.exec_module(module)
    return module


def _sample(script: ModuleType, platform: str, shift: float) -> object:
    n_cells = 40
    rng = np.random.default_rng(len(platform))
    astro = rng.random(n_cells) < 0.5 + shift
    labels = pd.DataFrame(
        {
            "total_counts": rng.integers(15, 200, n_cells),
            "ct_lineage_name": np.where(astro, "Astrocytes", "Neurons"),
            "ct_lineage_raw": np.full(n_cells, 0.9),
            "ct_broad_name": np.where(astro, "Astrocytes", "Neurons"),
            "ct_broad_raw": np.full(n_cells, 0.9),
            "mmc_whb_supercluster_name": np.where(
                astro, "Astrocyte", "Upper-layer intratelencephalic"
            ),
            "mmc_whb_supercluster_bp": np.full(n_cells, 0.9),
            "mmc_whb_supercluster_runner_up_1_name": np.full(n_cells, "Microglia"),
            "mmc_whb_supercluster_runner_up_1_bp": np.full(n_cells, 0.05),
        },
        index=pd.Index([f"{platform}{i}" for i in range(n_cells)], name="cell_id"),
    )
    sea = pd.DataFrame(
        {
            "broad": labels["ct_broad_name"].to_numpy(),
            "broad_raw": np.full(n_cells, 0.9),
        },
        index=labels.index,
    )
    scores = np.zeros((n_cells, len(E1_REFEREE_MARKERS)))
    classes = list(E1_REFEREE_MARKERS)
    scores[astro, classes.index("Astrocytes")] = 0.3
    scores[~astro, classes.index("Neurons")] = 0.3
    inputs = rule_inputs_from_provisional(labels, sea)
    rules = {
        vote: evaluate_human_rules(inputs, platform=platform, second_vote=vote)
        for vote in script.VOTES
    }
    return script.Sample(
        sample_id=f"PX_{platform}",
        platform=platform,
        labels=labels,
        sea=sea,
        xy=rng.random((n_cells, 2)) * 2000,
        legacy_broad=np.full(n_cells, "Neurons", dtype=object),
        scores=scores,
        aligned_frame=True,
        rules=rules,
    )


def test_sample_metrics_and_referee_rows(script: ModuleType) -> None:
    sample = _sample(script, "MERSCOPE", 0.0)
    row = script.sample_metrics(sample, "PX", "proseg_hybrid")
    assert row["n_table_cells"] == 40
    assert row["gate_level"] in {"full", "broad_only", "failed"}
    assert row["whb_sea_agree_ge20"] == pytest.approx(1.0)
    assert row["broad_cov_seaad"] == pytest.approx(1.0)
    assert row["below60_sea_rule_cost"] == pytest.approx(0.0)
    assert row["held_out"] is False
    referee = script.referee_rows(sample, "PX", "proseg_hybrid")
    legacy = next(
        item
        for item in referee
        if item["first"].startswith("new") and item["second"] == "legacy broad_class"
    )
    # The legacy label says Neurons everywhere; the astrocytes' markers side
    # with the new label.
    assert legacy["n_disputes"] > 0
    assert legacy["first_wins"] == pytest.approx(1.0)


def test_pair_jsd_without_a_mask_reports_the_whole_section(
    script: ModuleType, tmp_path: Path
) -> None:
    samples = {
        "MERSCOPE": _sample(script, "MERSCOPE", 0.0),
        "XENIUM": _sample(script, "XENIUM", 0.3),
    }
    jsd_rows, comp_rows = script.pair_jsd(
        samples, "PX", "proseg_hybrid", tmp_path, n_reps=20, tile_um=500.0, seed=0
    )
    kinds = {row["kind"] for row in jsd_rows}
    assert {"soft", "argmax", "confident", "soft_ge30"} <= kinds
    assert {row["region"] for row in jsd_rows} == {"whole_section"}
    for row in jsd_rows:
        assert row["ci_low"] <= row["jsd"] <= row["ci_high"]
    assert len(comp_rows) == 2 * len(jsd_rows)


def test_script_help_runs(script: ModuleType) -> None:
    with pytest.raises(SystemExit) as raised:
        script.main(["--help"])
    assert raised.value.code == 0
