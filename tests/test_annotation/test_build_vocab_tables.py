"""Tests for ``scripts/annotation/build_vocab_tables.py``.

The Allen taxonomy CSVs are not in the repository, so these tests rebuild a
minimal taxonomy with the same structure (terms, cluster membership, NT
annotations and cell counts) from the committed vocabularies and check that
the generator reproduces the committed tables from it and ``overrides.yaml``.
"""

from __future__ import annotations

import copy
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml  # type: ignore[import-untyped]

from merxen.annotation import vocab

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "annotation" / "build_vocab_tables.py"
GENERATED = (
    "whb_supercluster_vocab.csv",
    "seaad_mr_subclass_vocab.csv",
    "wmb_class_vocab.csv",
    "floors_human.csv",
    "floors_mouse.csv",
)
WHB_NT_TERMS = {
    "Excitatory": [("VGLUT1", 100)],
    "Inhibitory": [("GABA", 100)],
    # No class above half of the annotated cells.
    "Other": [("GABA", 40), ("VGLUT2", 40), ("HDC", 20)],
}
WMB_NT_TERMS = {"Excitatory": "Glut", "Inhibitory": "GABA", "Other": "Dopa"}
SEAAD_CLASS = {
    "Inhibitory": "Neuronal: GABAergic",
    "Excitatory": "Neuronal: Glutamatergic",
    "": "Non-neuronal and Non-neural",
}
SEAAD_VLMC_SUPERTYPES = ("VLMC_1", "VLMC_2-SEAAD", "Pericyte_1", "SMC-SEAAD")


@pytest.fixture(scope="module")
def generator() -> ModuleType:
    """Import the generator script as a module."""
    spec = importlib.util.spec_from_file_location("build_vocab_tables", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_whb(directory: Path) -> Path:
    table = vocab.load_asset_table("whb_supercluster_vocab.csv")
    terms, membership, clusters = [], [], []
    alias = 0
    for order, row in table.iterrows():
        terms.append(
            {
                "label": row["label"],
                "name": row["name"],
                "cluster_annotation_term_set_label": "CCN202210140_SUPC",
                "term_order": order,
            }
        )
        nt_terms = WHB_NT_TERMS.get(row["nt"], [(np.nan, 100)])
        for nt_term, cells in nt_terms:
            membership.append(
                {
                    "cluster_alias": alias,
                    "cluster_annotation_term_set_label": "CCN202210140_SUPC",
                    "cluster_annotation_term_set_name": "supercluster",
                    "cluster_annotation_term_name": row["name"],
                }
            )
            membership.append(
                {
                    "cluster_alias": alias,
                    "cluster_annotation_term_set_label": "CCN202210140_NEUR",
                    "cluster_annotation_term_set_name": "neurotransmitter",
                    "cluster_annotation_term_name": nt_term,
                }
            )
            clusters.append({"cluster_alias": alias, "number_of_cells": cells})
            alias += 1
    directory.mkdir(parents=True)
    pd.DataFrame(terms).to_csv(directory / "cluster_annotation_term.csv", index=False)
    pd.DataFrame(membership).to_csv(
        directory / "cluster_to_cluster_annotation_membership.csv", index=False
    )
    pd.DataFrame(clusters).to_csv(directory / "cluster.csv", index=False)
    return directory


def _write_seaad(path: Path) -> Path:
    table = vocab.load_asset_table("seaad_mr_subclass_vocab.csv")
    subclasses = table[table["supertype_prefix"] == ""].reset_index(drop=True)
    rows: list[dict[str, Any]] = [
        {
            "label": f"CLASS_{index}",
            "name": name,
            "cluster_annotation_term_set_label": "CCN20260630_LEVEL_0",
            "term_order": index,
            "parent_term_label": np.nan,
            "parent_term_name": np.nan,
        }
        for index, name in enumerate(SEAAD_CLASS.values())
    ]
    for order, row in subclasses.iterrows():
        rows.append(
            {
                "label": row["label"],
                "name": row["subclass"],
                "cluster_annotation_term_set_label": "CCN20260630_LEVEL_1",
                "term_order": order,
                "parent_term_label": np.nan,
                "parent_term_name": SEAAD_CLASS[row["nt"]],
            }
        )
        supertypes = (
            SEAAD_VLMC_SUPERTYPES
            if row["subclass"] == "VLMC & Perivascular"
            else (f"{row['subclass']}_1",)
        )
        for supertype in supertypes:
            rows.append(
                {
                    "label": f"SUPR_{supertype}",
                    "name": supertype,
                    "cluster_annotation_term_set_label": "CCN20260630_LEVEL_2",
                    "term_order": 0,
                    "parent_term_label": row["label"],
                    "parent_term_name": row["subclass"],
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_wmb(directory: Path) -> Path:
    table = vocab.load_asset_table("wmb_class_vocab.csv")
    terms, membership = [], []
    for order, row in table.iterrows():
        terms.append(
            {
                "label": row["label"],
                "name": row["class"],
                "cluster_annotation_term_set_label": "CCN20230722_CLAS",
                "term_order": order,
            }
        )
        entries = [
            ("CCN20230722_CLAS", "class", row["class"], 1),
            ("CCN20230722_NEUR", "neurotransmitter", WMB_NT_TERMS.get(row["nt"]), 1),
            ("CCN20230722_CLUS", "cluster", f"cluster_{order}", 250),
        ]
        for set_label, set_name, term, cells in entries:
            membership.append(
                {
                    "cluster_alias": order,
                    "cluster_annotation_term_set_label": set_label,
                    "cluster_annotation_term_set_name": set_name,
                    "cluster_annotation_term_name": term,
                    "number_of_cells": cells,
                }
            )
    directory.mkdir(parents=True)
    pd.DataFrame(terms).to_csv(directory / "cluster_annotation_term.csv", index=False)
    pd.DataFrame(membership).to_csv(
        directory / "cluster_to_cluster_annotation_membership.csv", index=False
    )
    return directory


@pytest.fixture
def taxonomy(tmp_path: Path) -> dict[str, Path]:
    """Write a minimal Allen-like taxonomy matching the committed tables."""
    return {
        "whb_taxonomy_dir": _write_whb(tmp_path / "whb"),
        "seaad_term_csv": _write_seaad(tmp_path / "seaad_terms.csv"),
        "wmb_taxonomy_dir": _write_wmb(tmp_path / "wmb"),
    }


def _overrides() -> dict[str, Any]:
    return yaml.safe_load((vocab.ASSET_DIR / "overrides.yaml").read_text())


def _write_overrides(tmp_path: Path, overrides: dict[str, Any]) -> Path:
    path = tmp_path / "overrides.yaml"
    path.write_text(yaml.safe_dump(overrides, sort_keys=False))
    return path


def test_generator_reproduces_the_committed_tables(
    generator: ModuleType, taxonomy: dict[str, Path]
) -> None:
    result = generator.build_all(
        overrides_path=vocab.ASSET_DIR / "overrides.yaml", **taxonomy
    )
    for name in GENERATED:
        committed = (vocab.ASSET_DIR / name).read_text()
        assert result.files[name] == committed, name
    notice = result.files["NOTICE"]
    assert "CC BY-NC 4.0" in notice
    assert "WHB-taxonomy 20240330" in notice
    assert "/media/" not in notice and "/srv/" not in notice
    assert len(result.inputs) == 6


def test_notice_depends_on_input_content_not_local_names(
    generator: ModuleType, taxonomy: dict[str, Path], tmp_path: Path
) -> None:
    """A renamed local copy of an input gives the same NOTICE (ABC paths)."""
    renamed = tmp_path / "SEA-AD-Multiregion-taxonomy__20260711__terms.csv"
    renamed.write_bytes(taxonomy["seaad_term_csv"].read_bytes())
    overrides = vocab.ASSET_DIR / "overrides.yaml"

    original = generator.build_all(overrides_path=overrides, **taxonomy)
    moved = generator.build_all(
        overrides_path=overrides, **{**taxonomy, "seaad_term_csv": renamed}
    )

    assert moved.files["NOTICE"] == original.files["NOTICE"]
    names = [item.name for item in original.inputs]
    assert names == [
        "WHB-taxonomy/20240330/cluster_annotation_term.csv",
        "WHB-taxonomy/20240330/cluster_to_cluster_annotation_membership.csv",
        "WHB-taxonomy/20240330/cluster.csv",
        "SEA-AD-Multiregion-taxonomy/20260711/cluster_annotation_term.csv",
        "WMB-taxonomy/20231215/cluster_annotation_term.csv",
        "WMB-taxonomy/20231215/cluster_to_cluster_annotation_membership.csv",
    ]
    for name in names:
        assert name in original.files["NOTICE"]


# Real Allen taxonomy inputs for the slow check of the committed tables.
REAL_TAXONOMY_ENV = {
    "--whb-taxonomy-dir": "MERXEN_WHB_TAXONOMY_DIR",
    "--seaad-term-csv": "MERXEN_SEAAD_TERM_CSV",
    "--wmb-taxonomy-dir": "MERXEN_WMB_TAXONOMY_DIR",
}


@pytest.mark.slow
def test_committed_tables_match_the_real_allen_taxonomies(
    generator: ModuleType,
) -> None:
    """``--check`` against the real Allen CSVs named by the environment."""
    paths = {flag: os.environ.get(name) for flag, name in REAL_TAXONOMY_ENV.items()}
    if not all(paths.values()) or not all(
        Path(str(path)).exists() for path in paths.values()
    ):
        pytest.skip(
            "set " + ", ".join(REAL_TAXONOMY_ENV.values()) + " to the Allen "
            "taxonomy inputs to check the committed tables against them"
        )
    argv = [part for flag, path in paths.items() for part in (flag, str(path))]

    assert generator.main([*argv, "--check"]) == 0


def test_committed_notice_names_every_generated_table() -> None:
    notice = (vocab.ASSET_DIR / "NOTICE").read_text()
    for name in (*GENERATED, "state_genes_human.csv", "heldout_markers_human.csv"):
        assert name in notice
    assert "Allen Institute Software License" in notice


def test_main_writes_then_checks(
    generator: ModuleType, taxonomy: dict[str, Path], tmp_path: Path
) -> None:
    output = tmp_path / "out"
    output.mkdir()
    argv = [
        "--whb-taxonomy-dir",
        str(taxonomy["whb_taxonomy_dir"]),
        "--seaad-term-csv",
        str(taxonomy["seaad_term_csv"]),
        "--wmb-taxonomy-dir",
        str(taxonomy["wmb_taxonomy_dir"]),
        "--output-dir",
        str(output),
    ]
    assert generator.main([*argv, "--check"]) == 1
    assert not any(output.iterdir())
    assert generator.main(argv) == 0
    assert generator.main([*argv, "--check"]) == 0
    assert (output / "whb_supercluster_vocab.csv").read_text() == (
        vocab.ASSET_DIR / "whb_supercluster_vocab.csv"
    ).read_text()


def test_missing_curated_mapping_fails(
    generator: ModuleType, taxonomy: dict[str, Path], tmp_path: Path
) -> None:
    overrides = _overrides()
    del overrides["whb"]["superclusters"]["Astrocyte"]
    with pytest.raises(generator.VocabBuildError, match="without a curated mapping"):
        generator.build_all(
            overrides_path=_write_overrides(tmp_path, overrides), **taxonomy
        )


def test_lineage_must_agree_with_nt_annotations(
    generator: ModuleType, taxonomy: dict[str, Path], tmp_path: Path
) -> None:
    overrides = copy.deepcopy(_overrides())
    overrides["whb"]["superclusters"]["Astrocyte"] = {
        "broad_class": "Neurons",
        "lineage": "Neurons",
    }
    with pytest.raises(generator.VocabBuildError, match="NT annotation"):
        generator.build_all(
            overrides_path=_write_overrides(tmp_path, overrides), **taxonomy
        )


def test_broad_class_must_sit_in_its_lineage(
    generator: ModuleType, taxonomy: dict[str, Path], tmp_path: Path
) -> None:
    overrides = _overrides()
    overrides["whb"]["superclusters"]["Oligodendrocyte"]["lineage"] = "Astrocytes"
    with pytest.raises(generator.VocabBuildError, match="does not contain"):
        generator.build_all(
            overrides_path=_write_overrides(tmp_path, overrides), **taxonomy
        )


def test_supertype_override_prefixes_must_cover_the_subclass(
    generator: ModuleType, taxonomy: dict[str, Path], tmp_path: Path
) -> None:
    overrides = _overrides()
    del overrides["seaad"]["supertype_overrides"]["VLMC & Perivascular"]["SMC"]
    with pytest.raises(generator.VocabBuildError, match="SMC-SEAAD"):
        generator.build_all(
            overrides_path=_write_overrides(tmp_path, overrides), **taxonomy
        )


def test_unknown_floor_class_fails(generator: ModuleType) -> None:
    spec = copy.deepcopy(_overrides()["floors"]["human"])
    spec["rows"].append(
        {"level": "broad", "floor_class": "Neurons", "MERSCOPE": 10, "XENIUM": 10}
    )
    with pytest.raises(generator.VocabBuildError, match="class 'Neurons'"):
        generator.build_human_floors(spec)


def test_majority_nt_needs_more_than_half(generator: ModuleType) -> None:
    terms = pd.Series(["GABA", "VGLUT1", np.nan])
    assert generator.majority_nt(
        terms, pd.Series([60.0, 40.0, 0.0]), min_share=0.5
    ) == ("Inhibitory", 1.0)
    nt, share = generator.majority_nt(
        terms, pd.Series([50.0, 50.0, 100.0]), min_share=0.5
    )
    assert nt == "Other"
    assert share == pytest.approx(0.5)
    assert generator.majority_nt(
        pd.Series([np.nan]), pd.Series([10.0]), min_share=0.5
    ) == (None, 0.0)
