"""Tests for RESOLVE's report flags (plan §4.3, §5.6; M4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from merxen.annotation import flags as fl
from merxen.annotation import shadow
from merxen.annotation.config import AnnotationFlagsConfig
from merxen.annotation.provenance import FlagProvenance, unsafe_structure_problems
from merxen.annotation.schema import Columns, column_specs

CLASSES = ("Neurons", "Astrocytes")


def test_shadow_reexports_the_production_primitives() -> None:
    assert shadow.contamination_flags is fl.contamination_flags
    assert shadow.expected_genes_quantile is fl.expected_genes_quantile
    assert shadow.CONTAMINATION_ALPHA == fl.CONTAMINATION_ALPHA == 0.01


# --------------------------------------------------------------------------
# Negative genes


def _negatives() -> pd.DataFrame:
    rows = [
        # class, gene, SEA-AD, WHB, stored
        ("Neurons", "G1", 0.001, 0.002, True),  # negative in both
        ("Neurons", "G2", 0.001, 0.2, False),  # negative in SEA-AD only
        ("Neurons", "G3", 0.001, np.nan, False),  # missing in WHB
        ("Neurons", "ENSG00000131095", 0.0, 0.0, False),  # GFAP: a state gene
        ("Astrocytes", "G2", 0.005, 0.009, False),  # stored disagrees
        ("Astrocytes", "G9", 0.0, 0.0, True),  # not a query gene
    ]
    return pd.DataFrame(
        {
            "broad_class": [row[0] for row in rows],
            "gene_id": [row[1] for row in rows],
            "detection_seaad_mr": [row[2] for row in rows],
            "n_cells_seaad_mr": 100.0,
            "detection_whb_frontal": [row[3] for row in rows],
            "n_cells_whb_frontal": 100.0,
            "is_state_gene": [row[1] == "ENSG00000131095" for row in rows],
            "negative": [row[4] for row in rows],
        }
    )


def test_negative_genes_need_both_references_and_exclude_state_genes() -> None:
    genes = ["G1", "G2", "G3", "ENSG00000131095"]
    negatives = fl.NegativeGeneSet.from_table(
        _negatives(),
        genes,
        classes=CLASSES,
        state_gene_ids=["ENSG00000131095"],
    )
    assert negatives.references == ("seaad_mr", "whb_frontal")
    assert negatives.genes("Neurons") == ["G1"]
    assert negatives.genes("Astrocytes") == ["G2"]
    assert negatives.genes("Microglia") == []
    assert negatives.n_disagree_with_stored == 1
    assert negatives.mask.shape == (2, 4)
    assert negatives.n_genes() == {"Neurons": 1, "Astrocytes": 1}


def test_negative_genes_fall_back_to_the_stored_column() -> None:
    table = _negatives().drop(columns=["detection_seaad_mr", "detection_whb_frontal"])
    negatives = fl.NegativeGeneSet.from_table(
        table,
        ["G1", "G2", "ENSG00000131095"],
        classes=CLASSES,
        state_gene_ids=["ENSG00000131095"],
    )
    assert negatives.references == ()
    assert negatives.genes("Neurons") == ["G1"]
    assert negatives.genes("Astrocytes") == []


# --------------------------------------------------------------------------
# Contamination


def _contamination_data(
    *, n_clean: int = 4000, n_planted: int = 400, planted_confident: bool = False
) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(0)
    n = n_clean + n_planted
    total = rng.integers(40, 400, n).astype(float)
    rate = rng.beta(2.0, 200.0, n)
    planted = np.zeros(n, dtype=bool)
    planted[n_clean:] = True
    rate[planted] = 0.30
    neg = rng.binomial(total.astype(int), rate).astype(float)
    classes = np.array(["Neurons"] * n, dtype=object)
    confident = ~planted | planted_confident
    return neg, total, classes, confident, planted


def test_contamination_flags_a_planted_spill_with_few_clean_flags() -> None:
    neg, total, classes, confident, planted = _contamination_data()
    result = fl.contamination_result(
        neg, total, classes, confident, platform="MERSCOPE", class_names=CLASSES
    )
    flag = result.flag.to_numpy(dtype=object, na_value=None)
    clean_rate = float(np.mean(flag[~planted] == True))  # noqa: E712
    detected = float(np.mean(flag[planted] == True))  # noqa: E712
    assert clean_rate <= 0.02
    assert detected >= 0.95
    fit = result.fits["Neurons"]
    assert fit is not None and fit.n_cells >= 900
    stratum = next(item for item in result.strata if item.cls == "Neurons")
    assert stratum.informative and stratum.informative_h16
    assert stratum.rate == pytest.approx(clean_rate, abs=0.01)
    # A class without cells has no null and a null flag.
    astro = next(item for item in result.strata if item.cls == "Astrocytes")
    assert not astro.informative and astro.reason == "no_null"
    score = result.score[~planted]
    assert np.nanmax(score) < 0.2


def test_contamination_needs_three_negative_counts() -> None:
    neg, total, classes, confident, _ = _contamination_data(n_planted=0)
    neg = np.append(neg, 2.0)
    total = np.append(total, 10.0)
    classes = np.append(classes, "Neurons")
    confident = np.append(confident, False)
    result = fl.contamination_result(
        neg, total, classes, confident, platform="XENIUM", class_names=CLASSES
    )
    assert result.p_value[-1] < 0.01
    assert not bool(result.flag[-1])
    loose = fl.contamination_result(
        neg,
        total,
        classes,
        confident,
        platform="XENIUM",
        class_names=CLASSES,
        min_neg_counts=2,
    )
    assert bool(loose.flag[-1])


def test_a_stratum_above_the_realised_rate_limit_is_uninformative() -> None:
    neg, total, classes, confident, planted = _contamination_data(
        n_clean=1000, n_planted=400, planted_confident=True
    )
    # Shallow contaminated cells are confident too; the deep null stays
    # clean, so 400 / 1400 of the confident calls flag: > 15%.
    total[~planted] = np.maximum(total[~planted], 200.0)
    total[planted] = 60.0
    neg[planted] = 18.0
    result = fl.contamination_result(
        neg, total, classes, confident, platform="XENIUM", class_names=("Neurons",)
    )
    stratum = result.strata[0]
    assert stratum.rate is not None and stratum.rate > 0.15
    assert not stratum.informative and not stratum.informative_h16
    assert stratum.reason is not None and "above_0.15" in stratum.reason
    assert result.flag.isna().all()
    assert not result.raw_flag.any()
    assert np.isfinite(result.score).all()
    # The switch itself, on a fixed raw flag.
    raw = np.array([True] * 20 + [False] * 80)
    members = np.ones(100, dtype=bool)
    switched = fl._stratum(
        "contaminated",
        "Neurons",
        "XENIUM",
        raw,
        members,
        members,
        max_informative=0.15,
    )
    assert not switched.informative and not switched.informative_h16
    kept = fl._stratum(
        "contaminated",
        "Neurons",
        "XENIUM",
        raw,
        members,
        members,
        max_informative=0.30,
    )
    assert kept.informative and not kept.informative_h16


def test_too_few_deep_confident_cells_leave_no_null() -> None:
    neg, total, classes, confident, _ = _contamination_data(n_clean=60, n_planted=0)
    result = fl.contamination_result(
        neg,
        total,
        classes,
        confident,
        platform="MERSCOPE",
        class_names=("Neurons",),
        min_null_cells=30,
    )
    assert result.fits["Neurons"] is None
    assert result.flag.isna().all()
    assert np.isfinite(result.score).all()


# --------------------------------------------------------------------------
# Diffuse profile


def _profile(n_genes: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.dirichlet(np.full(n_genes, 0.3))
    return values / values.sum()


def _multinomial_cells(
    profile: np.ndarray, n: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    depth = rng.integers(20, 300, n)
    draws = np.array([rng.multinomial(int(d), profile) for d in depth])
    return depth.astype(float), (draws > 0).sum(axis=1).astype(float)


def test_diffuse_flag_is_calibrated_on_multinomial_cells() -> None:
    profile = _profile(60, 1)
    depth, detected = _multinomial_cells(profile, 3000, 2)
    classes = np.array(["Neurons"] * 3000, dtype=object)
    result = fl.diffuse_result(
        depth,
        detected,
        classes,
        np.ones(3000, dtype=bool),
        {"Neurons": profile},
        platform="XENIUM",
        class_names=("Neurons",),
        n_simulations=200,
    )
    rate = float(result.flag.to_numpy(dtype=bool, na_value=False).mean())
    assert 0.005 <= rate <= 0.07
    assert result.strata[0].informative
    assert np.isfinite(result.expected).all()


def test_diffuse_cells_are_flagged_and_a_diffuse_stratum_is_uninformative() -> None:
    profile = _profile(60, 1)
    flat = np.full(60, 1 / 60)
    depth_a, detected_a = _multinomial_cells(profile, 1000, 3)
    depth_b, detected_b = _multinomial_cells(flat, 1000, 4)
    classes = np.array(["Neurons"] * 1000 + ["Astrocytes"] * 1000, dtype=object)
    result = fl.diffuse_result(
        np.concatenate([depth_a, depth_b]),
        np.concatenate([detected_a, detected_b]),
        classes,
        np.ones(2000, dtype=bool),
        {"Neurons": profile, "Astrocytes": profile},
        platform="XENIUM",
        class_names=CLASSES,
    )
    raw = result.raw_flag
    by_class = {item.cls: item for item in result.strata}
    assert by_class["Astrocytes"].rate is not None
    assert by_class["Astrocytes"].rate > 0.30
    assert not by_class["Astrocytes"].informative
    assert result.flag[1000:].isna().all()
    assert by_class["Neurons"].informative
    assert not raw[1000:].any()  # raw flags of the null stratum are cleared


def test_diffuse_without_a_profile_is_null() -> None:
    result = fl.diffuse_result(
        np.array([50.0, 60.0]),
        np.array([10.0, 12.0]),
        np.array(["Neurons", "Neurons"], dtype=object),
        np.ones(2, dtype=bool),
        {},
        platform="XENIUM",
        class_names=("Neurons",),
    )
    assert result.flag.isna().all()
    assert result.strata[0].reason == "no_profile"


# --------------------------------------------------------------------------
# OOD


def test_robust_z_and_ood_strata() -> None:
    rng = np.random.default_rng(5)
    corr = rng.normal(0.6, 0.05, 1200)
    corr[0] = 0.6 - 20 * 0.05
    classes = np.array(["Neurons"] * 1000 + ["Astrocytes"] * 200, dtype=object)
    bins = np.where(np.arange(1200) % 2 == 0, 30.0, 60.0)
    bins[1000:1020] = 120.0  # a stratum of 20 cells: no robust z
    result = fl.ood_result(
        corr,
        classes,
        bins,
        np.ones(1200, dtype=bool),
        platform="MERSCOPE",
        class_names=CLASSES,
    )
    assert bool(result.flag[0])
    assert result.z[0] < -10
    assert result.flag[1000:1020].isna().all()
    rate = float(result.raw_flag[1:1000].mean())
    assert rate < 0.01
    assert all(item.informative for item in result.strata)
    flat = fl.robust_z(np.full(40, 0.5))
    assert np.isnan(flat).all()
    assert np.isnan(fl.robust_z(np.arange(10.0))).all()


def test_discovery_caution_counts_null_flags_as_false() -> None:
    first = fl.nullable_flags(
        np.array([True, False, True]), np.array([True, True, False])
    )
    second = np.array([False, True, False])
    assert fl.discovery_caution([first, second]).tolist() == [True, True, False]
    with pytest.raises(ValueError, match="at least one"):
        fl.discovery_caution([])


def test_astro_lowcount_flags_mouse_astro_epen_below_100_counts() -> None:
    flags = fl.astro_lowcount_flags(
        np.array(
            ["30 Astro-Epen", "30 Astro-Epen", "01 IT-ET Glut", None], dtype=object
        ),
        np.array([50, 150, 50, 50]),
        np.array([True, True, True, True]),
    )
    assert flags.tolist() == [True, False, False, False]


# --------------------------------------------------------------------------
# compute_flags


def _flag_inputs(species: str = "human") -> fl.FlagInputs:
    rng = np.random.default_rng(8)
    n = 600
    genes = [f"G{index}" for index in range(30)]
    profile = _profile(30, 9)
    table = np.ones(n, dtype=bool)
    table[:20] = False
    depth = rng.integers(20, 300, n)
    matrix = np.array([rng.multinomial(int(d), profile) for d in depth])
    counts = matrix.sum(axis=1).astype(float) + 3
    rows = np.flatnonzero(table)
    classes = np.where(np.arange(n) % 2 == 0, "Neurons", "Astrocytes").astype(object)
    classes[25] = None
    negatives = pd.DataFrame(
        {
            "broad_class": ["Neurons", "Astrocytes"],
            "gene_id": ["G29", "G28"],
            "detection_seaad_mr": [0.0, 0.0],
            "detection_whb_frontal": [0.0, 0.0],
            "negative": [True, True],
        }
    )
    return fl.FlagInputs(
        species=species,
        platform="XENIUM",
        total_counts=counts,
        in_table=table,
        assigned_class=classes,
        confident=rng.uniform(size=n) < 0.7,
        method_disagree=np.arange(n) == 30,
        corr=rng.normal(0.6, 0.05, n),
        depth_bin=np.where(depth >= 60, 60.0, 30.0),
        exclude_hard=~table,
        query_counts=sparse.csr_matrix(matrix[rows]),
        query_rows=rows,
        gene_ids=tuple(genes),
        negatives=fl.NegativeGeneSet.from_table(negatives, genes, classes=CLASSES),
        profiles_by_class={"Neurons": profile, "Astrocytes": profile},
        wmb_class=(
            np.array(["30 Astro-Epen"] * n, dtype=object)
            if species == "mouse"
            else None
        ),
    )


def test_compute_flags_writes_every_flag_column_in_schema_dtypes() -> None:
    inputs = _flag_inputs()
    result = fl.compute_flags(inputs, AnnotationFlagsConfig(), class_names=CLASSES)
    specs = column_specs("human")
    expected = {
        Columns.CONTAMINATION_SCORE,
        Columns.NEG_COUNTS,
        Columns.FLAG_CONTAMINATED,
        Columns.EXPECTED_GENES_Q95,
        Columns.FLAG_DIFFUSE_PROFILE,
        Columns.OOD_Z,
        Columns.FLAG_OOD,
        Columns.MICROGLIA_STAT,
        Columns.MICROGLIA_WEIGHT,
        Columns.FLAG_MICROGLIAL_SPILLOVER,
        Columns.DISCOVERY_CAUTION,
    }
    assert set(result.columns) == expected
    frame = pd.DataFrame(result.columns)
    for name in expected:
        spec = specs[name]
        dtype = str(frame[name].dtype)
        allowed = {
            "float32": {"float32"},
            "nullable_int32": {"Int32", "int32"},
            "nullable_bool": {"boolean", "bool"},
            "bool": {"bool"},
        }[spec.kind]
        assert dtype in allowed, (name, dtype)
    outside = ~inputs.in_table
    assert frame.loc[outside, Columns.FLAG_CONTAMINATED].isna().all()
    assert frame.loc[outside, Columns.CONTAMINATION_SCORE].isna().all()
    assert not frame.loc[outside, Columns.DISCOVERY_CAUTION].any()
    assert bool(frame.loc[30, Columns.DISCOVERY_CAUTION])  # method disagreement
    assert frame.loc[25, Columns.FLAG_OOD] is pd.NA  # no class
    assert frame[Columns.FLAG_MICROGLIAL_SPILLOVER].isna().all()
    assert "OD-C5" in result.null_reasons["microglial_spillover"]
    assert result.gene_sets == {
        "negative_neurons": ["G29"],
        "negative_astrocytes": ["G28"],
    }
    provenance = result.provenance()
    assert isinstance(provenance, FlagProvenance)
    payload = provenance.model_dump(mode="json")
    assert unsafe_structure_problems(payload) == []
    assert set(payload["informative"]) == {"contaminated", "diffuse_profile", "ood"}
    assert "neurons__xenium" in payload["informative"]["contaminated"]
    assert payload["thresholds"]["h16_uninformative_above"] == 0.15
    rates = result.rates_frame()
    assert set(rates["flag"]) == {"contaminated", "diffuse_profile", "ood"}
    assert set(result.summary()) >= {"strata", "null_reasons", "contamination_null"}


def test_compute_flags_without_bundle_tables_is_null_with_reasons() -> None:
    inputs = _flag_inputs()
    bare = fl.FlagInputs(
        **{
            **inputs.__dict__,
            "negatives": None,
            "profiles_by_class": None,
        }
    )
    result = fl.compute_flags(bare, AnnotationFlagsConfig(), class_names=CLASSES)
    assert pd.Series(result.columns[Columns.FLAG_CONTAMINATED]).isna().all()
    assert pd.Series(result.columns[Columns.FLAG_DIFFUSE_PROFILE]).isna().all()
    assert "negative genes" in result.null_reasons["contaminated"]
    assert "profiles" in result.null_reasons["diffuse_profile"]
    assert result.columns[Columns.OOD_Z].dtype == np.float32


def test_compute_flags_keeps_the_mouse_hooks() -> None:
    result = fl.compute_flags(
        _flag_inputs("mouse"), AnnotationFlagsConfig(), class_names=CLASSES
    )
    assert Columns.FLAG_REGION_INCOHERENT in result.columns
    assert pd.Series(result.columns[Columns.FLAG_REGION_INCOHERENT]).isna().all()
    astro = result.columns[Columns.FLAG_ASTRO_LOWCOUNT]
    assert astro.dtype == bool and astro.any()
    assert result.null_reasons["microglial_spillover"] == fl.REASON_MOUSE_M6


def test_diffuse_flags_only_more_genes_than_the_quantile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fl, "expected_genes_quantile", lambda depth, labels, *a, **k: np.full(3, 5.0)
    )
    result = fl.diffuse_result(
        np.array([50.0, 50.0, 50.0]),
        np.array([4.0, 5.0, 6.0]),
        np.array(["Neurons"] * 3, dtype=object),
        np.ones(3, dtype=bool),
        {"Neurons": np.full(10, 0.1)},
        platform="XENIUM",
        class_names=("Neurons",),
        max_informative=1.0,
    )
    assert result.raw_flag.tolist() == [False, False, True]


def test_the_contamination_null_keeps_cells_tied_at_the_depth_cut() -> None:
    """The null holds confident cells at or above the top-quartile depth (§5.6).

    Integer counts tie at the 0.75 quantile: 10 cells at 100, 25 at 150 and
    5 at 200 put the cut at 150, and the 30 cells at or above it are fitted.
    """
    total = np.array([100.0] * 10 + [150.0] * 25 + [200.0] * 5)
    rng = np.random.default_rng(3)
    negative = rng.binomial(total.astype(int), 0.01).astype(np.float64)
    labels = np.array(["Neurons"] * len(total), dtype=object)
    confident = np.ones(len(total), dtype=bool)
    assert np.quantile(total, 0.75) == 150.0
    result = fl.contamination_flags(
        negative, total, labels, confident, ("Neurons",), min_null_cells=5
    )
    fit = result.fits["Neurons"]
    assert fit is not None and fit.n_cells == 30
