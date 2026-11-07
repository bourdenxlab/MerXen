"""Streaming MECR reference (Atera plan AT-8, WP3) against the in-memory path.

The streaming reference reads the atlas twice: pass 1 counts detections per
broad class over every panel gene without keeping the matrix, pass 2 keeps only
the detection candidates for the Wilcoxon test and the reference marker
co-occurrence. Markers and MECR values must equal the in-memory path; only
``p_value_adjusted`` (Benjamini-Hochberg over the candidates instead of every
panel gene) and the NaN Wilcoxon columns of non-candidate genes differ.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from statsmodels.stats.multitest import multipletests

from merxen.analysis import mecr
from merxen.analysis.mecr import (
    compute_mecr_pair_metrics,
    count_non_control_genes,
    detection_candidate_genes,
    discover_reference_markers,
    discover_streaming_reference_markers,
    load_atlas_panel_reference,
    resolve_reference_mode,
    run_mecr,
    run_mecr_reference,
    stream_atlas_panel_reference,
)
from merxen.config import MecrConfig, MecrReferenceConfig, MecrSampleConfig
from merxen.gene_sets import WHOLE_TRANSCRIPTOME_MIN_GENES

_CLASSES = ["Neurons", "Astrocytes", "Oligodendrocytes", "Microglia"]
# Taxonomy names that collapse to the classes above, plus one class that is
# not a target (its cells are dropped by the loader).
_TAXONOMY = {
    "Neurons": "Upper-layer intratelencephalic",
    "Astrocytes": "Astrocyte",
    "Oligodendrocytes": "Oligodendrocyte",
    "Microglia": "Microglia",
    "Fibroblasts": "Fibroblast",
}
_EXACT_COLUMNS = [
    "gene",
    "broad_class",
    "target_fraction",
    "other_fraction",
    "n_target_cells",
    "n_other_cells",
    "passes_detection_thresholds",
    "qualifying_class_count",
    "is_unique_marker",
]


def _write_atlas(tmp_path: Path) -> dict[str, Any]:
    """Two expression shards, taxonomy metadata and a panel of 64 symbols."""
    rng = np.random.default_rng(7)
    n_genes = 70
    symbols = [f"Gene{index:02d}" for index in range(n_genes)]
    symbols[69] = "Gene05"  # A duplicated symbol (summed by the loader).
    var = pd.DataFrame(
        {"gene_symbol": symbols},
        index=[f"ENSG{index:05d}" for index in range(n_genes)],
    )
    shard_classes = {
        "neurons": ["Neurons"] * 260,
        "nonneurons": (
            ["Astrocytes"] * 90
            + ["Oligodendrocytes"] * 110
            + ["Microglia"] * 70
            + ["Fibroblasts"] * 12
        ),
    }
    metadata_rows = []
    paths = []
    for shard, classes in shard_classes.items():
        n_cells = len(classes)
        labels = np.asarray(classes)
        # Ubiquitous background on genes 40-59, sparse noise elsewhere.
        counts = np.zeros((n_cells, n_genes), dtype=np.float32)
        counts[:, 40:60] = rng.poisson(1.2, size=(n_cells, 20))
        counts[:, 60:69] = rng.poisson(0.02, size=(n_cells, 9))
        for class_index, broad_class in enumerate(_CLASSES):
            rows = np.flatnonzero(labels == broad_class)
            if not len(rows):
                continue
            # Five specific genes per class at decreasing detection rates
            # (some around the 25% threshold), absent elsewhere.
            for offset, rate in enumerate((0.8, 0.6, 0.4, 0.26, 0.2)):
                gene = class_index * 10 + offset
                hit = rows[rng.random(len(rows)) < rate]
                counts[hit, gene] = rng.integers(1, 6, size=len(hit))
            # A gene leaking at ~0.5% into other classes (still eligible) and
            # one at ~3% (not eligible).
            counts[rows[: len(rows) // 2], class_index * 10 + 5] = 2
            counts[rows[: len(rows) // 2], class_index * 10 + 6] = 2
        others = np.flatnonzero(labels != "Neurons")
        counts[others[:1], 5] += 1  # 1 / 270 others: 0.37% < 1%.
        counts[others[:9], 6] += 1  # 9 / 270 others: 3.3% > 1%.
        # Gene07 qualifies for Astrocytes and Oligodendrocytes (ambiguous).
        for broad_class in ("Astrocytes", "Oligodendrocytes"):
            rows = np.flatnonzero(labels == broad_class)
            counts[rows[: int(0.6 * len(rows))], 7] = 3
        # The duplicated Gene05 column adds neuron counts to Gene05.
        neurons = labels == "Neurons"
        counts[neurons, 69] = rng.poisson(0.3, size=int(neurons.sum()))
        cell_labels = [f"{shard}_cell_{index}" for index in range(n_cells)]
        path = tmp_path / f"{shard}.h5ad"
        ad.AnnData(
            X=sparse.csr_matrix(counts),
            obs=pd.DataFrame(index=cell_labels),
            var=var,
        ).write_h5ad(path)
        paths.append(path)
        metadata_rows.extend(
            {"cell_label": label, "cluster_alias": broad_class}
            for label, broad_class in zip(cell_labels, classes, strict=True)
        )
    cell_metadata_path = tmp_path / "cell_metadata.csv"
    pd.DataFrame(metadata_rows).to_csv(cell_metadata_path, index=False)
    taxonomy_path = tmp_path / "taxonomy.csv"
    pd.DataFrame(
        {
            "label": [f"label_{key}" for key in _TAXONOMY],
            "name": list(_TAXONOMY.values()),
            "cluster_annotation_term_set_label": ["SUPC"] * len(_TAXONOMY),
        }
    ).to_csv(taxonomy_path, index=False)
    membership_path = tmp_path / "membership.csv"
    pd.DataFrame(
        {
            "cluster_annotation_term_label": [f"label_{key}" for key in _TAXONOMY],
            "cluster_annotation_term_set_label": ["SUPC"] * len(_TAXONOMY),
            "cluster_alias": list(_TAXONOMY),
        }
    ).to_csv(membership_path, index=False)
    panel = [*sorted(set(symbols[:64])), "MissingGene", "NegControlProbe_00001"]
    return {
        "paths": paths,
        "cell_metadata_path": cell_metadata_path,
        "taxonomy_metadata_path": taxonomy_path,
        "cluster_membership_path": membership_path,
        "panel": panel,
    }


def _config(
    tmp_path: Path,
    atlas: dict[str, Any],
    *,
    mode: str,
    workers: int = 1,
    platform: str = "XENIUM",
    name: str = "out",
) -> MecrReferenceConfig:
    return MecrReferenceConfig(
        output_dir=tmp_path / name,
        samples=[
            MecrSampleConfig(
                sample_id="sample",
                platform=platform,  # type: ignore[arg-type]
                zarr_path=tmp_path / "unused.zarr",
            )
        ],
        reference_h5ad_paths=atlas["paths"],
        cell_metadata_path=atlas["cell_metadata_path"],
        taxonomy_metadata_path=atlas["taxonomy_metadata_path"],
        cluster_membership_path=atlas["cluster_membership_path"],
        taxonomy_level="SUPC",
        target_broad_classes=_CLASSES,
        reference_chunk_rows=37,
        reference_mode=mode,  # type: ignore[arg-type]
        reference_workers=workers,
    )


@pytest.fixture
def atlas(tmp_path: Path) -> dict[str, Any]:
    return _write_atlas(tmp_path)


def _bh(p_values: np.ndarray) -> np.ndarray:
    return np.asarray(multipletests(p_values, alpha=0.05, method="fdr_bh")[1])


def test_streaming_matches_in_memory_markers_and_mecr(
    tmp_path: Path, atlas: dict[str, Any]
) -> None:
    """Same markers, fractions and reference MECR; documented p_adj change."""
    config = _config(tmp_path, atlas, mode="streaming")
    reference = load_atlas_panel_reference(config, panel_genes=atlas["panel"])
    stats_mem, markers_mem = discover_reference_markers(
        reference,
        class_key="broad_class",
        target_classes=_CLASSES,
    )
    streamed = stream_atlas_panel_reference(config, panel_genes=atlas["panel"])
    stats_str, markers_str = discover_streaming_reference_markers(streamed)

    # The fixture exercises every rule: ambiguous, leaking and boundary genes.
    assert set(markers_mem["broad_class"]) == set(_CLASSES)
    assert "Gene07" not in set(markers_mem["gene"])
    assert "Gene05" in set(markers_mem["gene"])
    assert "Gene06" not in set(markers_mem["gene"])
    assert stats_mem.shape == stats_str.shape
    pd.testing.assert_frame_equal(
        markers_str[_EXACT_COLUMNS], markers_mem[_EXACT_COLUMNS]
    )
    pd.testing.assert_frame_equal(stats_str[_EXACT_COLUMNS], stats_mem[_EXACT_COLUMNS])

    # Detection counts cover every panel gene; Wilcoxon only the candidates.
    candidates = detection_candidate_genes(
        streamed.detection, min_target_fraction=0.25, max_other_fraction=0.01
    )
    assert streamed.candidates is not None
    assert list(streamed.candidates.var_names) == candidates
    assert len(candidates) < reference.n_vars
    tested = stats_str["gene"].isin(candidates)
    assert stats_str.loc[~tested, "wilcoxon_score"].isna().all()
    assert stats_str.loc[~tested, "p_value_adjusted"].isna().all()
    for column in ("wilcoxon_score", "p_value"):
        np.testing.assert_allclose(
            stats_str.loc[tested, column].to_numpy(),
            stats_mem.loc[tested, column].to_numpy(),
            rtol=1e-12,
        )
    np.testing.assert_allclose(
        stats_str.loc[tested, "logfoldchange"].to_numpy(),
        stats_mem.loc[tested, "logfoldchange"].to_numpy(),
        rtol=1e-5,
        atol=1e-6,
    )
    # pvals_adj: BH over the candidates (streaming) vs every panel gene.
    for broad_class in _CLASSES:
        rows = stats_str[(stats_str["broad_class"] == broad_class) & tested]
        np.testing.assert_allclose(
            rows["p_value_adjusted"].to_numpy(),
            _bh(rows["p_value"].to_numpy()),
            rtol=1e-12,
        )
        mem_rows = stats_mem[stats_mem["broad_class"] == broad_class]
        np.testing.assert_allclose(
            mem_rows["p_value_adjusted"].to_numpy(),
            _bh(mem_rows["p_value"].to_numpy()),
            rtol=1e-12,
        )

    # The reference marker co-occurrence (and so MECR) is identical.
    pairs_mem = compute_mecr_pair_metrics(reference, markers_mem)
    pairs_str = compute_mecr_pair_metrics(streamed.candidates, markers_str)
    assert len(pairs_mem) > 0
    pd.testing.assert_frame_equal(pairs_str, pairs_mem)


def test_streaming_never_stacks_the_panel_matrix(
    tmp_path: Path, atlas: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only candidate columns are ever stacked; the in-memory loader is unused."""
    config = _config(tmp_path, atlas, mode="streaming")
    stacked: list[tuple[int, int]] = []
    real_vstack = sparse.vstack

    def spy_vstack(blocks: Any, *args: Any, **kwargs: Any) -> Any:
        result = real_vstack(blocks, *args, **kwargs)
        stacked.append(result.shape)
        return result

    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("streaming called the in-memory loader")

    monkeypatch.setattr(mecr.sparse, "vstack", spy_vstack)
    monkeypatch.setattr(mecr, "load_atlas_panel_reference", refuse)
    streamed = stream_atlas_panel_reference(config, panel_genes=atlas["panel"])

    n_panel = len(streamed.detection.genes)
    assert streamed.candidates is not None
    n_candidates = streamed.candidates.n_vars
    assert n_candidates < n_panel
    assert stacked, "pass 2 stacks the candidate chunks"
    assert all(n_cols == n_candidates for _rows, n_cols in stacked), stacked


def test_streaming_detection_ignores_chunking_and_workers(
    tmp_path: Path, atlas: dict[str, Any]
) -> None:
    """Pass-1 counts do not depend on chunk size or the worker processes."""
    serial = _config(tmp_path, atlas, mode="streaming")
    pooled = _config(tmp_path, atlas, mode="streaming", workers=2)
    pooled.reference_chunk_rows = 101
    first = stream_atlas_panel_reference(serial, panel_genes=atlas["panel"])
    second = stream_atlas_panel_reference(pooled, panel_genes=atlas["panel"])
    np.testing.assert_array_equal(first.detection.detected, second.detection.detected)
    np.testing.assert_array_equal(
        first.detection.class_sizes, second.detection.class_sizes
    )
    # Retained cells only: the 12 non-target cells are dropped.
    assert first.detection.class_sizes.tolist() == [260, 90, 110, 70]
    assert first.candidates is not None and second.candidates is not None
    assert (first.candidates.X != second.candidates.X).nnz == 0


def test_run_mecr_reference_outputs_match_between_modes(
    tmp_path: Path, atlas: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Markers and reference pair CSVs are byte-identical; the manifest says why."""
    monkeypatch.setattr(
        mecr, "collect_spatial_panel_genes", lambda _samples: list(atlas["panel"])
    )
    outputs = {}
    for mode in ("in_memory", "streaming"):
        config = _config(tmp_path, atlas, mode=mode, name=mode)
        outputs[mode] = run_mecr_reference(config)
    for key in ("reference_pairs_csv", "panel_genes_csv"):
        assert (
            outputs["streaming"][key].read_bytes()
            == outputs["in_memory"][key].read_bytes()
        ), key
    # The marker rows agree; only p_value_adjusted (BH over the candidates)
    # and float noise in logfoldchange differ.
    markers = {
        mode: pd.read_csv(paths["markers_csv"]) for mode, paths in outputs.items()
    }
    pd.testing.assert_frame_equal(
        markers["streaming"][[*_EXACT_COLUMNS, "wilcoxon_score", "p_value"]],
        markers["in_memory"][[*_EXACT_COLUMNS, "wilcoxon_score", "p_value"]],
    )
    manifests = {
        mode: json.loads(paths["manifest"].read_text())
        for mode, paths in outputs.items()
    }
    streaming = manifests["streaming"]
    assert streaming["reference_mode"]["requested"] == "streaming"
    assert streaming["reference_mode"]["resolved"] == "streaming"
    assert (
        streaming["wilcoxon"]["n_tested_genes"]
        < (manifests["in_memory"]["wilcoxon"]["n_tested_genes"])
    )
    assert manifests["in_memory"]["wilcoxon"]["genes"] == "all panel genes"
    for key in (
        "n_reference_cells",
        "n_reference_panel_genes",
        "reference_class_counts",
        "n_unique_markers",
        "n_reference_marker_pairs",
    ):
        assert streaming[key] == manifests["in_memory"][key], key


def test_auto_mode_streams_only_whole_transcriptome_panels(tmp_path: Path) -> None:
    """auto streams above WHOLE_TRANSCRIPTOME_MIN_GENES non-control genes."""
    config = MecrReferenceConfig(
        output_dir=tmp_path,
        samples=[
            MecrSampleConfig(
                sample_id="sample",
                platform="ATERA",
                zarr_path=tmp_path / "unused.zarr",
            )
        ],
    )
    assert config.reference_mode == "auto"
    assert config.reference_workers == 1
    limit = WHOLE_TRANSCRIPTOME_MIN_GENES
    genes = [f"G{index}" for index in range(limit)]
    controls = [f"NegControlProbe_{index:05d}" for index in range(3)]
    assert count_non_control_genes(genes + controls) == limit
    assert resolve_reference_mode(config, genes + controls)[0] == "in_memory"
    mode, reason = resolve_reference_mode(config, [*genes, "G_extra"])
    assert mode == "streaming"
    assert f"{limit + 1:,} non-control panel genes" in reason
    config.reference_mode = "in_memory"
    assert resolve_reference_mode(config, [*genes, "G_extra"]) == (
        "in_memory",
        "requested",
    )


def test_atera_barnyard_draws_every_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ATERA barnyard plots never subsample; XENIUM keeps its subsample."""
    rng = np.random.default_rng(3)
    counts = np.zeros((600, 2), dtype=np.int64)
    counts[:300, 0] = rng.integers(0, 4, size=300)
    counts[300:, 1] = rng.integers(0, 4, size=300)
    counts[0, :] = 1
    adata = ad.AnnData(
        X=sparse.csr_matrix(counts),
        var=pd.DataFrame(index=["SLC17A7", "GFAP"]),
    )
    markers_path = tmp_path / "markers.csv"
    pd.DataFrame(
        {"gene": ["SLC17A7", "GFAP"], "broad_class": ["Neurons", "Astrocytes"]}
    ).to_csv(markers_path, index=False)
    monkeypatch.setattr(mecr, "load_spatial_count_table", lambda _s: adata.copy())

    choices: list[int] = []
    real_rng = np.random.default_rng

    class SpyRng:
        def __init__(self, seed: int) -> None:
            self._rng = real_rng(seed)

        def choice(self, values: Any, size: int, replace: bool) -> Any:
            choices.append(size)
            return self._rng.choice(values, size=size, replace=replace)

    monkeypatch.setattr(mecr.np.random, "default_rng", SpyRng)
    for platform in ("ATERA", "XENIUM"):
        choices.clear()
        config = MecrConfig(
            pair_id="PAIR1",
            segmentation="original_seg",
            output_dir=tmp_path / platform,
            samples=[
                MecrSampleConfig(
                    sample_id=f"PAIR1_{platform}",
                    platform=platform,  # type: ignore[arg-type]
                    zarr_path=tmp_path / "unused.zarr",
                )
            ],
            reference_markers_path=markers_path,
            barnyard_max_points=100,
        )
        outputs = run_mecr(config)
        plots = sorted(outputs["barnyard_plot_dir"].glob("*.png"))
        assert plots, platform
        assert (outputs[f"{platform.lower()}_pairs_csv"]).exists()
        if platform == "ATERA":
            assert choices == []
        else:
            assert choices == [100]


def test_spatial_tables_are_read_without_the_whole_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One table is read directly; the SpatialData reader is the fallback."""
    import spatialdata as sd

    counts = sparse.csr_matrix(np.array([[1, 0, 2], [0, 3, 0]], dtype=np.float32))
    tables = {
        key: ad.AnnData(
            X=counts * (index + 1),
            obs=pd.DataFrame(index=["c1", "c2"]),
            var=pd.DataFrame(index=[f"{key}_A", f"{key}_B", "SHARED"]),
        )
        for index, key in enumerate(("table_original", "table_other"))
    }
    zarr_path = tmp_path / "store.zarr"
    sd.SpatialData(tables=tables).write(zarr_path)
    sample = MecrSampleConfig(
        sample_id="S", platform="ATERA", zarr_path=zarr_path, table_key="table_other"
    )
    via_sdata = sd.read_zarr(zarr_path).tables["table_other"]

    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("read the whole store")

    with monkeypatch.context() as patch:
        patch.setattr(sd, "read_zarr", refuse)
        genes = mecr.collect_spatial_panel_genes([sample])
        table = mecr.load_spatial_count_table(sample)
    assert genes == ["SHARED", "table_other_A", "table_other_B"]
    assert list(table.var_names) == list(via_sdata.var_names)
    assert (table.X != via_sdata.X).nnz == 0
    # A missing table still fails through SpatialData's message.
    missing = sample.model_copy(update={"table_key": "table_missing"})
    with pytest.raises(KeyError, match="table_missing"):
        mecr.load_spatial_count_table(missing)
