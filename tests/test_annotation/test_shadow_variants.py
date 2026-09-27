"""Tests for the shadow variant re-maps (published queries, held-out, X1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from merxen.annotation.mapmycells_engine import read_tidy_parquet
from merxen.annotation.shadow import (
    map_query_variant,
    published_queries,
    whb_labels_from_tidy,
)

from .conftest import FakeMmc
from .test_pipeline import GENE_IDS, _human_bundles, _pair_samples, _panel


def test_published_queries_rebuild_the_map_query(tmp_path: Path) -> None:
    samples = _pair_samples(tmp_path / "published", source="clustered")
    panel = _panel(GENE_IDS[:5])
    queries = published_queries(
        {sample.platform: sample.h5ad_path for sample in samples},
        panel,
        species="human",
    )
    assert set(queries) == {"MERSCOPE", "XENIUM"}
    merscope = queries["MERSCOPE"]
    # Every published object is a table cell; the Blank control is removed.
    assert len(merscope.cell_ids) == 8
    assert merscope.gene_ids == GENE_IDS[:5]
    assert merscope.total_counts[7] == 30
    assert queries["XENIUM"].counts.shape == (3, 5)


def test_map_query_variant_drops_and_rescales_genes(
    tmp_path: Path, fake_mmc: FakeMmc
) -> None:
    samples = _pair_samples(tmp_path / "published", source="clustered")
    panel = _panel(GENE_IDS)
    bundles = _human_bundles(fake_mmc, panel)
    whb = bundles[("whb_frontal_supc_clus", panel.panel_hash)]
    query = published_queries(
        {sample.platform: sample.h5ad_path for sample in samples},
        panel,
        species="human",
    )["MERSCOPE"]
    output = map_query_variant(
        query,
        whb,
        tmp_path / "out" / "variant.parquet",
        work_dir=tmp_path / "work",
        drop_gene_ids=[GENE_IDS[0]],
        log2_factors={GENE_IDS[1]: 1.0},
        n_processors=2,
        run_metadata={"note": "test"},
    )
    tidy, metadata = read_tidy_parquet(output)
    assert metadata["variant_dropped_genes"] == [GENE_IDS[0]]
    assert metadata["variant_rescaled"] is True
    assert metadata["n_query_genes"] == len(GENE_IDS) - 1
    command = fake_mmc.calls[-1]["command"]
    assert "--query_markers.serialized_lookup" in command
    lookup = Path(command[command.index("--query_markers.serialized_lookup") + 1])
    assert GENE_IDS[0] not in lookup.read_text()
    vocab = pd.read_csv(whb.vocab_snapshot, dtype=str, keep_default_na=False)
    total = pd.Series(query.total_counts, index=pd.Index(query.cell_ids))
    labels = whb_labels_from_tidy(tidy, vocab, total, level=whb.levels[0])
    # c0 had only GEXC (dropped) and GOTHER; c1 is now astrocyte-dominated.
    assert labels.loc["M1", "mmc_whb_supercluster_name"] == "Astrocyte"
    assert labels.loc["M7", "ct_broad_name"] == "Oligodendrocytes"
    assert np.isfinite(labels["total_counts"]).all()
    reused = map_query_variant(
        query, whb, output, work_dir=tmp_path / "work", drop_gene_ids=GENE_IDS
    )
    assert reused == output
    with pytest.raises(ValueError, match="every query gene"):
        map_query_variant(
            query,
            whb,
            tmp_path / "out" / "empty.parquet",
            work_dir=tmp_path / "work",
            drop_gene_ids=GENE_IDS,
        )
