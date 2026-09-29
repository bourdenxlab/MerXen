"""Slow: MAP + RESOLVE on E7's 30k-cell ag7 subset with the real WMB store (M6).

The subset is E7's (``exp/E7/queries/ag7_obs_30k.csv``) of the published ag7
proseg_hybrid clustered H5AD. MAP runs the region step and the pruned re-map
with the dwight store; RESOLVE applies the mouse rules, gate and flags. The
test skips when the store, the ctm environment or the inputs are absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import pandas as pd
import pytest
from click.testing import CliRunner

from merxen.annotation.pipeline import read_label_table
from merxen.annotation.schema import validate_label_table
from merxen.cli import main as cli_main

STORE = Path("/media/mathieubo/SSD1/MerXen/annotation_references")
GENE_TABLE = Path(
    "/media/mathieubo/SSD1/MerXen/mapmycells/abc_atlas/metadata/WMB-10X/20241115/"
    "gene.csv"
)
AG7 = Path(
    "/srv/storage/Araceli-ageing/results/ag7/proseg_hybrid/clustering_squidpy/"
    "clustering_squidpy_out/merscope/ag7_MERSCOPE_clustered.h5ad"
)
SUBSET = Path(
    "/srv/storage/MerXen/annotation_dev/evidence_20260926/exp/E7/queries/"
    "ag7_obs_30k.csv"
)
MANUAL_REGIONS = {"Isocortex", "OLF", "HPF", "CTXsp", "TH", "HY", "MB"}
E3_MICROGLIA_GENES = {"Csf1r", "Cx3cr1", "Aif1", "Blnk", "C1qa"}


@pytest.mark.slow
def test_ag7_30k_subset_maps_prunes_and_resolves(tmp_path: Path) -> None:
    if not all(path.exists() for path in (STORE, GENE_TABLE, AG7, SUBSET)):
        pytest.skip("the dwight store or the ag7 inputs are not available")
    pytest.importorskip("cell_type_mapper")
    ids = pd.read_csv(SUBSET, usecols=["obs_id"])["obs_id"].astype(str)
    full = ad.read_h5ad(AG7, backed="r")
    subset = full[full.obs_names.isin(ids)].to_memory()
    full.file.close()
    assert subset.n_obs == 30_000
    h5ad = tmp_path / "ag7_30k" / "merscope" / "ag7_MERSCOPE_clustered.h5ad"
    h5ad.parent.mkdir(parents=True)
    subset.write_h5ad(h5ad)
    map_dir = tmp_path / "map"
    runner = CliRunner()

    mapped = runner.invoke(
        cli_main,
        [
            "annotate",
            "--from-clustered-h5ad",
            str(h5ad),
            "--species",
            "mouse",
            "--store",
            str(STORE),
            "--out",
            str(map_dir),
            "--work-dir",
            str(tmp_path / "work"),
            "--n-processors",
            "4",
            "--no-reuse",
            "--gene-id-fallback-csv",
            str(GENE_TABLE),
        ],
    )
    assert mapped.exit_code == 0, mapped.output
    manifest = json.loads((map_dir / "map_manifest.json").read_text())
    (record,) = manifest["samples"].values()
    regions = record["mouse_regions"]
    assert regions["status"] == "pruned"
    assert set(regions["inferred_regions"]) == MANUAL_REGIONS  # MO3 at 30k (E7 §4)
    assert len(regions["nodes_to_drop"]) == 60  # E7 drop_hybrid_manual.txt

    resolved = runner.invoke(
        cli_main,
        [
            "annotate-resolve",
            "--map-dir",
            str(map_dir),
            "--out",
            str(tmp_path / "resolve"),
            "--gene-id-fallback-csv",
            str(GENE_TABLE),
            "--mouse-g4-sections",
            "C57BL6J-638850.31,C57BL6J-638850.32,C57BL6J-638850.33",
        ],
    )
    assert resolved.exit_code == 0, resolved.output
    summary = json.loads(
        next((tmp_path / "resolve").glob("*resolve_summary.json")).read_text()
    )
    (sample,) = summary["samples"].values()
    gate = sample["mouse_gate"]
    assert gate["level"] == "full"  # MO7: ag7 not failed (G1 not given: warns)
    assert gate["signal_status"]["g1"] == "not_evaluated"
    assert gate["signal_status"]["g2"] in ("pass", "warn")
    assert gate["signal_status"]["g4"] == "pass"
    assert gate["signals"]["g3_implausible_share"] < 0.03
    flags = sample["spillover_checks"]
    assert flags["n_genes"] == 5
    assert flags["astrocyte_fpr"]["fpr"] <= 0.005  # MO5
    assert set(sample["flags"]["thresholds"]) >= {"microglia_stat_min"}
    f1 = sample["region_coherence"]["rate"]
    assert 0.002 <= f1 <= 0.010  # MO6 band (E7 30k after pruning: 0.68%)
    levels = sample["resolution"]["levels"]
    assert levels["class"]["confident_share_table"] > 0.6
    labels_path = next(
        (tmp_path / "resolve" / "merscope").glob("*_celltype_labels.parquet")
    )
    labels, _ = read_label_table(labels_path)
    validate_label_table(labels, "mouse", h5ad_index=labels["cell_id"].astype(str))
    derived = json.loads(
        next(
            (tmp_path / "resolve" / "merscope").glob("*_annotation_manifest.json")
        ).read_text()
    )["flags"]["gene_sets"]["microglial_spillover_genes"]
    symbols = pd.read_csv(GENE_TABLE).set_index("gene_identifier")["gene_symbol"]
    assert {symbols[gene] for gene in derived} == E3_MICROGLIA_GENES
