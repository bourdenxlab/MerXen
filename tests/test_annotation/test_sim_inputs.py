"""Tests for the simulation-input assets and recipes (plan §4.7, §8.3 v7.4-v7.5)."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from merxen.annotation import sim_inputs as si
from merxen.annotation.resolvability import gene_efficiency

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "data"
PCP2 = "ENSMUSG00000004630"  # weak tier, log2 factor -3 (cerebellar; composition)
PRESENT = "ENSMUSG00000000058"  # informative, top class Vascular (11.7% of cells)
ABSENT_TOP = "ENSMUSG00000000001"  # informative, top class OEC (0% of cells)


@pytest.fixture(scope="module")
def registry() -> dict[str, si.SimInputAsset]:
    return si.load_registry()


@pytest.fixture(scope="module")
def mouse_table(registry: dict[str, si.SimInputAsset]) -> pd.DataFrame:
    return si.efficiency_table(registry[si.EFFICIENCY_MOUSE_PRIME])


# --------------------------------------------------------------------------
# Registry and provenance


def test_registry_holds_the_m3c_assets_with_complete_provenance(
    registry: dict[str, si.SimInputAsset],
) -> None:
    assert set(registry) == {
        si.EFFICIENCY_MOUSE_PRIME,
        si.PROFILE_MOUSE_PRIME_FF,
        si.STRESS_HUMAN_LUNG,
        si.SCENARIO_HUMAN_LUNG,
        si.PRIME_PANEL_LISTS,
    }
    script = REPO_ROOT / "scripts" / "annotation" / "build_sim_inputs.py"
    script_sha = hashlib.sha256(script.read_bytes()).hexdigest()
    for asset in registry.values():
        assert asset.trust_effect == "none"
        assert asset.size < si.MAX_ASSET_BYTES
        for key in si.REQUIRED_PROVENANCE:
            assert asset.provenance.get(key), (asset.asset_id, key)
        assert asset.provenance["licence_note"] == si.LICENCE_NOTE
        assert asset.provenance["deriving_script_sha256"] == script_sha
        for item in asset.provenance["source_files"]:
            assert item["url"].startswith("https://")
            assert len(item["sha256"]) == 64
        for item in asset.provenance["derived_from"]:
            assert len(item["sha256"]) == 64
    labels = {asset.asset_id: asset.label for asset in registry.values()}
    assert labels[si.EFFICIENCY_MOUSE_PRIME] == (
        "XOA 3.0 vendor segmentation, public 10x section"
    )
    assert "cross-tissue" in labels[si.STRESS_HUMAN_LUNG]
    profile = registry[si.PROFILE_MOUSE_PRIME_FF].provenance
    assert profile["cross_check"]["sha256"].startswith("ff35db2e")


def test_notices_attribute_the_public_data() -> None:
    notice = (si.SIM_INPUTS_DIR / si.NOTICE_FILE).read_text(encoding="utf-8")
    assert "CC BY 4.0" in notice and "10x Genomics" in notice
    assert si.LICENCE_NOTE in notice
    main = (si.SIM_INPUTS_DIR.parent / "NOTICE").read_text(encoding="utf-8")
    assert "sim_inputs/" in main and "CC BY 4.0" in main


def _copy_assets(tmp_path: Path) -> Path:
    target = tmp_path / "sim_inputs"
    shutil.copytree(si.SIM_INPUTS_DIR, target)
    return target


def test_an_altered_asset_is_refused(tmp_path: Path) -> None:
    directory = _copy_assets(tmp_path)
    path = directory / f"{si.STRESS_HUMAN_LUNG}.csv"
    path.write_text(path.read_text() + "ENSG00000000000,0\n")
    with pytest.raises(si.SimInputError, match="sha256"):
        si.load_registry(directory)


def test_an_incomplete_provenance_is_refused(tmp_path: Path) -> None:
    directory = _copy_assets(tmp_path)
    sidecar = directory / f"{si.STRESS_HUMAN_LUNG}{si.SIDECAR_SUFFIX}"
    payload = json.loads(sidecar.read_text())
    del payload["provenance"]["licence_note"]
    sidecar.write_text(json.dumps(payload))
    with pytest.raises(si.SimInputError, match="licence_note"):
        si.load_registry(directory)


def test_an_asset_cannot_claim_a_trust_effect(tmp_path: Path) -> None:
    directory = _copy_assets(tmp_path)
    sidecar = directory / f"{si.STRESS_HUMAN_LUNG}{si.SIDECAR_SUFFIX}"
    payload = json.loads(sidecar.read_text())
    payload["trust_effect"] = "promote"
    sidecar.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="trust_effect"):
        si.load_registry(directory)


def test_member_tables_never_cross_species_or_chemistry(
    registry: dict[str, si.SimInputAsset],
) -> None:
    mouse = si.member_table_for("mouse", "xenium_prime", "wmb_10xv3", registry)
    assert mouse is not None and mouse.asset_id == si.EFFICIENCY_MOUSE_PRIME
    assert si.member_table_for("human", "xenium_prime", "whb", registry) is None
    assert si.member_table_for("mouse", "unknown", "wmb_10xv3", registry) is None
    assert si.member_table_for("mouse", "merscope", "wmb_10xv3", registry) is None


def test_asset_hashes_record_each_sha256(registry: dict[str, si.SimInputAsset]) -> None:
    hashes = si.asset_hashes(registry.values())
    assert list(hashes) == sorted(registry)
    for asset_id, record in hashes.items():
        assert record["sha256"] == registry[asset_id].sha256
        assert record["asset_version"] == 1


# --------------------------------------------------------------------------
# R3_measured_HO


def test_all_measured_rule_reproduces_the_d3_vector(mouse_table: pd.DataFrame) -> None:
    """Pre-registration §21 (ii-a): max |delta| <= 1e-9 on every 5K gene."""
    fixture = pd.read_csv(DATA / "m3c_d3_efficiency_mouse5k.csv")
    result = si.r3_efficiency(
        fixture["gene_id"].tolist(), mouse_table, rule="all_measured", seed=0
    )
    delta = np.abs(result.log2_efficiency - fixture["log2_efficiency"].to_numpy())
    assert len(fixture) == 5006
    assert float(delta.max()) <= 1e-9
    summary = result.summary()
    assert summary["frac_below_0.25"] == pytest.approx(0.0867, abs=5e-4)
    assert summary["frac_below_0.125"] == pytest.approx(0.0204, abs=5e-4)


def test_restricted_rule_measures_informative_present_genes_only(
    mouse_table: pd.DataFrame,
) -> None:
    genes = [PRESENT, ABSENT_TOP, PCP2, "ENSMUSG99999999999"]
    result = si.r3_efficiency(genes, mouse_table, rule="restricted", seed=0)
    source = dict(zip(result.genes, result.source, strict=True))
    assert source == {
        PRESENT: "measured",
        ABSENT_TOP: "resampled",
        PCP2: "resampled",
        "ENSMUSG99999999999": "resampled",
    }
    raw = dict(zip(result.genes, result.log2_raw, strict=True))
    factor = float(mouse_table.loc[PRESENT, "log2_factor"])
    z = si.keyed_normal(0, si.STREAM_MEASURED_RESIDUAL, PRESENT)
    assert raw[PRESENT] == pytest.approx(factor + 0.20 * z, abs=1e-12)
    mask = si.measured_mask(mouse_table)
    pool = np.sort(np.clip(mouse_table.loc[mask, "log2_factor"].astype(float), -3, 3))
    assert int(mask.sum()) == result.n_pool == 1898
    for gene in (ABSENT_TOP, PCP2, "ENSMUSG99999999999"):
        u = si.keyed_uniform(0, si.STREAM_MEASURED_RESAMPLE, gene)
        assert raw[gene] == pool[math.floor(u * len(pool))]
    assert float(np.median(result.efficiency)) == pytest.approx(1.0)


def _synthetic_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tier": ["informative", "informative", "informative", "weak", "weak"],
            "log2_factor": [5.0, -7.0, 0.5, -3.0, -2.0],
            "expected_counts": [5000.0, 5000.0, 999.0, 500.0, 5000.0],
            "observed_counts": [9000, 10, 400, 3, 1200],
            "top_class": ["A", "A", "A", "B", "B"],
            "top_class_section_share": [0.2, 0.004999, 0.3, 0.5, 0.5],
        },
        index=pd.Index(
            ["G_cap", "G_rare_top", "G_low_expected", "G_weak", "G_weak_deep"]
        ),
    )


def test_restricted_rule_caps_at_three_log2_and_needs_both_minimums() -> None:
    table = _synthetic_table()
    result = si.r3_efficiency(list(table.index), table, rule="restricted", seed=3)
    source = dict(zip(result.genes, result.source, strict=True))
    assert source == {
        "G_cap": "measured",
        "G_rare_top": "resampled",
        "G_low_expected": "resampled",
        "G_weak": "resampled",
        "G_weak_deep": "resampled",
    }
    z = si.keyed_normal(3, si.STREAM_MEASURED_RESIDUAL, "G_cap")
    assert result.log2_raw[0] == pytest.approx(3.0 + 0.20 * z, abs=1e-12)
    # The pool is the capped measured factors only (here the single gene).
    assert np.all(result.log2_raw[1:] == 3.0)


def test_restricted_rule_is_keyed_by_gene_id(mouse_table: pd.DataFrame) -> None:
    genes = sorted(mouse_table.index)[:400]
    base = si.r3_efficiency(genes, mouse_table, rule="restricted", seed=0)
    more = si.r3_efficiency(
        [*genes, "ENSMUSG99999999999"], mouse_table, rule="restricted", seed=0
    )
    shuffled = list(reversed(genes))
    permuted = si.r3_efficiency(shuffled, mouse_table, rule="restricted", seed=0)
    np.testing.assert_array_equal(base.log2_raw, more.log2_raw[: len(genes)])
    np.testing.assert_array_equal(base.log2_raw, permuted.log2_raw[::-1])
    # Adding a gene only rescales the whole vector (median normalisation).
    ratio = more.efficiency[: len(genes)] / base.efficiency
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-12)
    other = si.r3_efficiency(genes, mouse_table, rule="restricted", seed=1)
    assert not np.array_equal(base.log2_raw, other.log2_raw)
    again = si.r3_efficiency(genes, mouse_table, rule="restricted", seed=0)
    np.testing.assert_array_equal(base.efficiency, again.efficiency)


def test_an_unknown_table_rule_is_refused(mouse_table: pd.DataFrame) -> None:
    with pytest.raises(si.SimInputError, match="rule"):
        si.r3_efficiency([PRESENT], mouse_table, rule="subclass")


# --------------------------------------------------------------------------
# R1_xtissue_lung_stress


def test_xtissue_stress_multiplies_r1_by_the_lung_ratio(
    registry: dict[str, si.SimInputAsset],
) -> None:
    ratio = si.ratio_table(registry[si.STRESS_HUMAN_LUNG])
    measured_gene = str(ratio.index[0])
    genes = [measured_gene, "ENSG99999999998", "ENSG99999999999"]
    base = gene_efficiency(len(genes), 0.8, 0)
    efficiency, measured = si.xtissue_stress_efficiency(genes, base, ratio, seed=0)
    assert measured.tolist() == [True, False, False]
    z = [si.keyed_normal(0, si.STREAM_XTISSUE, gene) for gene in genes[1:]]
    ln = np.log(base) + np.array(
        [float(ratio.iloc[0]) * math.log(2.0), 0.702 * z[0], 0.702 * z[1]]
    )
    expected = np.exp(ln) / np.median(np.exp(ln))
    np.testing.assert_allclose(efficiency, expected, rtol=1e-12)
    assert float(ratio.abs().max()) <= 2.0


# --------------------------------------------------------------------------
# Depth profiles


def test_mouse_profile_falls_back_to_the_lineage_pools(
    registry: dict[str, si.SimInputAsset],
) -> None:
    profile = si.profile_from_asset(registry[si.PROFILE_MOUSE_PRIME_FF])
    assert profile.n_cells == 63147
    assert profile.source("01 IT-ET Glut") == "01 IT-ET Glut"
    assert profile.source("03 OB-CR Glut") == si.POOL_NEURONAL  # < 100 cells
    assert profile.source("15 HY Gnrh1 Glut") == si.POOL_NEURONAL  # absent
    assert profile.source("32 OEC") == si.POOL_NON_NEURONAL
    neuronal = profile.pool(si.POOL_NEURONAL)
    glia = profile.pool(si.POOL_NON_NEURONAL)
    assert len(neuronal) + len(glia) == profile.n_cells
    assert float(np.median(profile.by_class["30 Astro-Epen"])) == 628.0
    assert float(np.median(profile.pool(si.POOL_ALL))) == 1089.0
    rng = np.random.default_rng(5)
    draws = profile.sample("01 IT-ET Glut", 4, rng)
    rng = np.random.default_rng(5)
    values = profile.by_class["01 IT-ET Glut"]
    np.testing.assert_array_equal(draws, values[rng.integers(0, len(values), 4)])


def test_the_lung_scenario_is_pooled(registry: dict[str, si.SimInputAsset]) -> None:
    scenario = si.profile_from_asset(registry[si.SCENARIO_HUMAN_LUNG])
    assert scenario.pooled
    assert scenario.n_cells == 275556
    assert scenario.source("Exc") == si.POOL_ALL
    assert scenario.median("Astro") == 245.0


def test_depth_profile_csv_per_class_and_pooled(tmp_path: Path) -> None:
    per_class = tmp_path / "per_class.csv"
    pd.DataFrame(
        {
            "cell_id": [f"c{i}" for i in range(250)],
            "total_counts": [100] * 120 + [900] * 130,
            "class": ["01 IT-ET Glut"] * 120 + ["30 Astro-Epen"] * 130,
        }
    ).to_csv(per_class, index=False)
    profile = si.read_depth_profile_table(per_class, species="mouse")
    assert not profile.pooled
    assert profile.source("01 IT-ET Glut") == "01 IT-ET Glut"
    assert profile.source("02 NP-CT-L6b Glut") == si.POOL_NEURONAL
    assert profile.source("34 Immune") == si.POOL_NON_NEURONAL
    assert profile.median("34 Immune") == 900.0
    pooled = tmp_path / "pooled.csv"
    pd.DataFrame({"depth": [10, 20], "weight": [1, 3]}).to_csv(pooled, index=False)
    flat = si.read_depth_profile_table(pooled, species="human")
    assert flat.pooled and flat.n_cells == 4
    assert flat.source("Exc") == si.POOL_ALL
    shares = flat.bin_shares([10, 20])
    assert shares[si.POOL_ALL] == {0: 0.0, 10: 0.25, 20: 0.75}


def test_neuronal_classes_follow_the_vocabularies() -> None:
    assert si.is_neuronal_class("01 IT-ET Glut", "mouse") is True
    assert si.is_neuronal_class("31 OPC-Oligo", "mouse") is False
    assert si.is_neuronal_class("99 Unknown", "mouse") is None
    assert si.is_neuronal_class("Exc", "human") is True
    assert si.is_neuronal_class("Astro", "human") is False


# --------------------------------------------------------------------------
# Chemistry


def test_chemistry_resolution() -> None:
    lists = si.prime_panel_lists()
    mouse = sorted(lists["xenium_prime_5k_mouse"][1])
    human = sorted(lists["xenium_prime_5k_human"][1])
    assert len(mouse) == 5006 and len(human) == 5001
    resolved = si.resolve_chemistry(mouse, species="mouse", platform="XENIUM")
    assert resolved.chemistry == "xenium_prime"
    assert resolved.panel_key == "xenium_prime_5k_mouse"
    near = mouse[: int(len(mouse) * 0.96)]
    assert si.resolve_chemistry(near, species="mouse", platform=None).chemistry == (
        "xenium_prime"
    )
    assert (
        si.resolve_chemistry(mouse[:500], species="mouse", platform="XENIUM").chemistry
        == "unknown"
    )
    # A list never resolves a panel of the other species.
    assert (
        si.resolve_chemistry(mouse, species="human", platform="XENIUM").chemistry
        == "unknown"
    )
    assert si.resolve_chemistry(
        human, species="human", platform="XENIUM"
    ).chemistry == ("xenium_prime")
    assert (
        si.resolve_chemistry(mouse, species="mouse", platform="MERSCOPE").chemistry
        == "merscope"
    )
    declared = si.resolve_chemistry(
        mouse[:10], species="mouse", platform="XENIUM", declared="xenium_prime"
    )
    assert declared.chemistry == "xenium_prime" and declared.reason == "declared"
