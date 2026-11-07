"""Unit tests of the expression items' helpers (``report_expression``; items 4, 7)."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from merxen.annotation import report_expression as rx


def test_broad_profiles_are_reference_cell_weighted_means() -> None:
    rows = pd.DataFrame(
        {
            "node_name": ["Upper-layer intratelencephalic"] * 2
            + ["Deep-layer intratelencephalic"] * 2
            + ["Astrocyte"] * 2,
            "n_cells": [1, 1, 3, 3, 5, 5],
            "gene_id": ["g1", "g2"] * 3,
            "detection_fraction": [0.2, 0.0, 0.6, 0.4, 0.9, 0.1],
            "mean_log2cpm": [1.0, 0.0, 3.0, 2.0, 5.0, 1.0],
            "mean_cpm": [10.0, 0.0, 30.0, 20.0, 50.0, 5.0],
        }
    )
    broad_of = {
        "Upper-layer intratelencephalic": "Neurons",
        "Deep-layer intratelencephalic": "Neurons",
        "Astrocyte": "Astrocytes",
    }
    out = rx.broad_profiles(rows, broad_of)
    assert out is not None
    neurons = out[out["node_name"] == "Neurons"].set_index("gene_id")
    # (1 x 0.2 + 3 x 0.6) / 4 = 0.5; the mean over all reference neurons.
    assert neurons.loc["g1", "detection_fraction"] == pytest.approx(0.5)
    assert neurons.loc["g2", "mean_cpm"] == pytest.approx(15.0)
    assert neurons.loc["g1", "n_cells"] == 4
    astro = out[out["node_name"] == "Astrocytes"].set_index("gene_id")
    assert astro.loc["g1", "mean_log2cpm"] == pytest.approx(5.0)
    assert rx.broad_profiles(rows, {}) is None


def test_canonical_marker_genes_take_rank_one_and_alternates_per_class() -> None:
    symbols = ["AQP4", "GJA1", "P2RY12", "CX3CR1", "OPALIN", "XYZ"]
    genes = [f"ENSG{index}" for index in range(len(symbols))]
    picked = rx.canonical_marker_genes("human", genes, symbols)
    by_symbol = dict(zip(genes, symbols, strict=True))
    names = [by_symbol[gene] for gene in picked]
    # Astrocytes and microglia by their rank-1 markers (P2RY12 even though
    # it is poorly detected on MERSCOPE: the dotplot shows its absence); the
    # oligodendrocytes by the rank-2 alternate, as no rank-1 marker is present.
    assert names == ["AQP4", "GJA1", "OPALIN", "P2RY12"]
    assert "CX3CR1" not in names and "XYZ" not in names
    assert rx.canonical_marker_genes("human", ["ENSG1"], ["aqp4"]) == ["ENSG1"]


def test_cross_platform_basis_follows_the_scope() -> None:
    same = rx.cross_platform_basis(
        SimpleNamespace(
            panel_mode="intersection",
            jsd_purpose="annotation",
            kinds=("soft", "soft_ge30", "confident", "argmax"),
        )
    )
    assert (same.per_platform, same.prefix, same.variant) == (False, "mmc_whb", "set_a")
    assert same.label_kind == "confident"
    per = rx.cross_platform_basis(
        SimpleNamespace(
            panel_mode=None,
            jsd_purpose="intersection_xpanel",
            kinds=("soft", "soft_ge30", "argmax"),
        )
    )
    assert per.per_platform and per.prefix == "mmc_whb_xpanel"
    assert per.variant == "intersection_run" and per.label_kind == "argmax"
    assert "confident" not in per.kinds
