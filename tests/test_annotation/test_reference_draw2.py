"""Tests for gate P's second disjoint WMB test draw (M13b C20a; plan §14).

``build_wmb_selfmap_testset`` with ``resolvability.gate_p_wmb_draw`` 2 builds
draw 2 as its own ``wmb_selfmap_testset`` bundle: disjoint from the marker
training sample, the self-map list and draw 1, at most
``gate_p_topup_max_cluster_frac`` of any cluster's 10Xv3 cells, each class
topped up to ``gate_p_class_min_test_cells`` pooled with draw 1's check half,
with the availability per class recorded. Draw 1 and every production
``build_hash`` stay unchanged. Synthetic WMB inputs (the taxonomy of
``test_reference``), no MapMyCells.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from merxen.annotation import reference
from merxen.annotation import resolvability as res
from merxen.annotation.config import (
    KNOWN_REFERENCES,
    AnnotationConfig,
    AnnotationReferenceSpec,
)
from merxen.annotation.reference import (
    builder_for,
    prepare_reference_spec,
    sample_wmb_training_cells,
)
from merxen.annotation.store import (
    BUNDLE_MANIFEST_NAME,
    LargePanelRefusedError,
    ReferenceStore,
    build_hash_payload,
    compute_build_hash,
)

from .test_reference import (
    CLAS,
    REAL_CLASSES,
    REAL_SUBCLASS_CLUSTERS,
    SUPT,
    make_panel,
    write_abc_taxonomy,
)
from .test_resolvability_v6_golden import _fake_source, golden_for_alias_state

N_GENES = 40
DONORS = ("D1", "D2", "D3")
NEURONAL_CLASS = "CS20230722_CLAS_01"
GLIAL_CLASS = "CS20230722_CLAS_30"


def write_wmb_draw_sources(
    tmp_path: Path, *, n_per_cluster: int = 60, n_listed: int = 3
) -> dict[str, Path]:
    """Synthetic WMB inputs: 9 clusters x ``n_per_cluster`` 10Xv3 cells.

    The taxonomy is ``test_reference``'s (two classes, four subclasses and
    supertypes, nine clusters); each cluster's first ``n_listed`` cells form
    the self-map list, and every cell has an ABC-style ``donor_label``.
    """
    import anndata as ad
    import scipy.sparse as sp

    rng = np.random.default_rng(11)
    genes = [f"ENSMUSG{index:011d}" for index in range(1, N_GENES + 1)]
    subclass_of = {
        alias: subclass
        for subclass, aliases in REAL_SUBCLASS_CLUSTERS.items()
        for alias in aliases
    }
    class_of = {
        subclass: label
        for label, (_name, subclasses) in REAL_CLASSES.items()
        for subclass in subclasses
    }
    subclass_index = {name: index for index, name in enumerate(REAL_SUBCLASS_CLUSTERS)}
    class_index = {label: index for index, label in enumerate(REAL_CLASSES)}
    rows: list[dict[str, Any]] = []
    blocks = []
    for alias in range(1, 10):
        base = rng.gamma(2.0, 1.0, N_GENES)
        base[alias - 1] += 40.0
        base[10 + subclass_index[subclass_of[alias]]] += 40.0
        base[20 + class_index[class_of[subclass_of[alias]]]] += 60.0
        blocks.append(rng.poisson(base, size=(n_per_cluster, N_GENES)))
        matrix = "WMB-10Xv3-AAA" if alias <= 4 else "WMB-10Xv3-BBB"
        rows.extend(
            {
                "cell_label": f"cell-{alias}-{index}",
                "feature_matrix_label": matrix,
                "library_method": "10Xv3",
                "cluster_alias": alias,
                "donor_label": DONORS[index % len(DONORS)],
            }
            for index in range(n_per_cluster)
        )
    meta = pd.DataFrame(rows)
    counts = np.vstack(blocks).astype(np.float32)
    h5ad_dir = tmp_path / "WMB-10Xv3"
    h5ad_dir.mkdir()
    for matrix, group in meta.groupby("feature_matrix_label"):
        ad.AnnData(
            X=sp.csr_matrix(counts[group.index.to_numpy()]),
            obs=pd.DataFrame(index=group["cell_label"].to_numpy()),
            var=pd.DataFrame(index=genes),
        ).write_h5ad(h5ad_dir / f"{matrix}-raw.h5ad")
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    meta.to_csv(metadata / "cell_metadata.csv", index=False)
    write_abc_taxonomy(metadata)
    listed = meta.groupby("cluster_alias").head(n_listed)
    truth = tmp_path / "truth.csv"
    listed[["cell_label"]].assign(subclass="x").to_csv(truth, index=False)
    return {
        "h5ad_dir": h5ad_dir,
        "metadata": metadata,
        "truth": truth,
        "genes": Path(json.dumps(genes)),
    }


def draw_config(**resolvability: Any) -> AnnotationConfig:
    """A version-6 mouse config (as MS1 and MS2) with small draw constants."""
    settings: dict[str, Any] = {
        "version": 6,
        "mouse_nonneuronal_extra_per_supertype": 2,
        "gate_p_class_min_test_cells": 40,
        "gate_p_topup_max_cluster_frac": 0.3,
    }
    settings.update(resolvability)
    return AnnotationConfig(species="mouse", resolvability=settings)


def with_draw(config: AnnotationConfig, draw: int) -> AnnotationConfig:
    """The config of a draw (only ``gate_p_wmb_draw`` changes)."""
    return config.model_copy(
        update={
            "resolvability": config.resolvability.model_copy(
                update={"gate_p_wmb_draw": draw}
            )
        }
    )


def wmb_draw_spec(sources: Mapping[str, Path]) -> AnnotationReferenceSpec:
    return prepare_reference_spec(
        AnnotationReferenceSpec(
            reference_id=reference.WMB_TESTSET_REFERENCE_ID,
            species="mouse",
            role="resolvability",
            max_cells_per_cluster=20,
            sources={
                "wmb_h5ad_dir": sources["h5ad_dir"],
                "wmb_metadata_dir": sources["metadata"],
                "wmb_selfmap_test_cells": sources["truth"],
            },
        )
    )


def build(
    store: ReferenceStore,
    spec: AnnotationReferenceSpec,
    panel: Any,
    config: AnnotationConfig,
) -> tuple[Path, dict[str, Any]]:
    bundle = store.get_or_build(
        spec, panel, builder=builder_for(spec, config), config=config
    )
    manifest = json.loads((Path(bundle.path) / BUNDLE_MANIFEST_NAME).read_text())
    return Path(bundle.path), manifest


def test_the_second_draw_is_disjoint_capped_and_records_its_availability(
    tmp_path: Path,
) -> None:
    sources = write_wmb_draw_sources(tmp_path)
    genes = json.loads(str(sources["genes"]))
    panel = make_panel(genes[:30], species="mouse")
    spec = wmb_draw_spec(sources)
    config = draw_config()
    store = ReferenceStore(tmp_path / "store")
    first_dir, first = build(store, spec, panel, config)
    second_dir, second = build(store, spec, panel, with_draw(config, 2))
    # Its own bundle: draw 2 enters the build hash, draw 1's payload has none.
    assert first_dir != second_dir
    first_params = first["build_hash_payload"]["builder_params"]
    second_params = second["build_hash_payload"]["builder_params"]
    assert "gate_p_draw" not in first_params
    rule = second_params["gate_p_draw"]
    assert rule == reference.wmb_second_draw_rule(with_draw(config, 2))
    assert {
        key: value for key, value in second_params.items() if key != "gate_p_draw"
    } == (first_params)
    test_set = second["builder_output"]["test_set"]
    record = test_set["gate_p_draw"]
    assert test_set["draw"] == 2
    one = res.load_test_cells(first_dir)
    two = res.load_test_cells(second_dir)
    first_cells = {str(cell) for cell in one.obs.index}
    second_cells = {str(cell) for cell in two.obs.index}
    # Disjoint from draw 1, the self-map list and the marker-training sample.
    listed = set(pd.read_csv(sources["truth"])["cell_label"])
    sampled, _ = sample_wmb_training_cells(
        sources["metadata"] / "cell_metadata.csv",
        {"WMB-10Xv3-AAA": Path(), "WMB-10Xv3-BBB": Path()},
        max_cells_per_cluster=20,
        exclude_cells=listed,
    )
    assert second_cells and not second_cells & first_cells
    assert not second_cells & listed
    assert not second_cells & set(sampled["cell_label"])
    assert record["disjoint_from_first_draw"] and record["disjoint_from_training"]
    assert record["disjoint_from_selfmap_list"]
    assert test_set["marker_training_sample"]["disjoint"] is True
    # The draw-1 digest is that of the draw-1 bundle's cells.
    assert record["first_draw"]["cell_digest"] == reference.test_cell_digest(
        first_cells
    )
    assert record["cell_digest"] == reference.test_cell_digest(second_cells)
    # At most 30% of any cluster (the fixture's cap; 5% by default).
    clusters = two.obs[f"{res.TRUTH_PREFIX}CCN20230722_CLUS"].astype(str)
    assert clusters.value_counts().max() <= math.floor(0.3 * 60)
    assert record["max_cluster_share_taken"] <= 0.3
    # The parts: <= 10 per supertype, the non-neuronal extra, the top-up.
    sources_col = two.obs[reference.TEST_SOURCE_COLUMN].astype(str)
    per_supertype = two.obs[sources_col == reference.DRAW2_SOURCE_SUPERTYPE]
    assert (
        per_supertype[f"{res.TRUTH_PREFIX}{SUPT}"].value_counts().max()
        <= reference.WMB_SELFMAP_PER_SUPERTYPE
    )
    extra = two.obs[sources_col == reference.DRAW2_SOURCE_NONNEURONAL]
    assert set(extra[f"{res.TRUTH_PREFIX}{CLAS}"]) == {GLIAL_CLASS}
    assert extra[f"{res.TRUTH_PREFIX}{SUPT}"].value_counts().max() <= 2
    # Each class reaches 40 pooled with draw 1's check half.
    check_half = one.obs[res.cell_split_half(list(one.obs.index)) == 1]
    check = check_half[f"{res.TRUTH_PREFIX}{CLAS}"].value_counts()
    drawn = two.obs[f"{res.TRUTH_PREFIX}{CLAS}"].value_counts()
    for cls in (NEURONAL_CLASS, GLIAL_CLASS):
        item = record["per_class"][cls]
        assert item["n_draw1_check_half"] == int(check.get(cls, 0))
        assert item["n_draw2"] == int(drawn[cls])
        assert item["n_pooled"] == int(check.get(cls, 0)) + int(drawn[cls])
        assert item["n_pooled"] >= 40 and item["reaches_target"]
        assert item["n_reference_10xv3"] == (4 if cls == NEURONAL_CLASS else 5) * 60
        assert item["n_available_under_cap"] >= item["n_draw2"]
    assert record["classes_short_of_target"] == []
    assert record["pooled_share_in_classes_reaching_target"] == 1.0
    assert sources_col.eq(reference.DRAW2_SOURCE_TOP_UP).any()
    # Panel-restricted counts, as draw 1.
    assert two.genes == one.genes == sorted(genes[:30])
    # Draw 1 is unchanged by the code that builds draw 2: the same hash again.
    again = store.prepare_request(
        spec, panel, builder=builder_for(spec, config), config=config
    ).build_hash
    assert again == first_dir.name


def test_the_second_draw_is_reproducible_and_records_classes_that_run_out(
    tmp_path: Path,
) -> None:
    sources = write_wmb_draw_sources(tmp_path)
    genes = json.loads(str(sources["genes"]))
    panel = make_panel(genes[:30], species="mouse")
    spec = wmb_draw_spec(sources)
    # A version-7 family: draw 1 holds its own class top-up (v7.6), which
    # draw 2 leaves out like every other draw-1 cell.
    config = with_draw(draw_config(version="auto", gate_p_class_min_test_cells=500), 2)
    store = ReferenceStore(tmp_path / "one")
    first, manifest = build(store, spec, panel, config)
    second, _ = build(ReferenceStore(tmp_path / "two"), spec, panel, config)
    assert first.name == second.name
    pd.testing.assert_frame_equal(
        res.load_test_cells(first).obs, res.load_test_cells(second).obs
    )
    draw_1, _ = build(store, spec, panel, with_draw(config, 1))
    one = res.load_test_cells(draw_1).obs
    assert (one[reference.TEST_SOURCE_COLUMN] == reference.TOP_UP_SOURCE).any()
    assert not set(one.index) & set(res.load_test_cells(first).obs.index)
    test_set = manifest["builder_output"]["test_set"]
    assert test_set["resolvability_version"] == 7
    assert test_set["first_draw_class_top_up"]["n_cells"] > 0
    record = test_set["gate_p_draw"]
    assert record["first_draw"]["cell_digest"] == reference.test_cell_digest(one.index)
    # 500 pooled is beyond what the caps leave: recorded, never exceeded.
    assert record["classes_reaching_target"] == []
    for item in record["per_class"].values():
        assert not item["reaches_target"] and item["ran_out"]
        assert item["n_draw2"] <= item["n_available_under_cap"]


def test_the_primary_refuses_the_second_draw_before_any_build(tmp_path: Path) -> None:
    sources = write_wmb_draw_sources(tmp_path, n_per_cluster=25)
    genes = json.loads(str(sources["genes"]))
    spec = prepare_reference_spec(
        AnnotationReferenceSpec(
            reference_id="wmb_panel",
            species="mouse",
            role="primary",
            max_cells_per_cluster=20,
            sources={
                "wmb_h5ad_dir": sources["h5ad_dir"],
                "wmb_metadata_dir": sources["metadata"],
                "wmb_mapping_stats": sources["truth"],
                "wmb_selfmap_test_cells": sources["truth"],
            },
        )
    )
    config = with_draw(draw_config(), 2)
    store = ReferenceStore(tmp_path / "store")
    with pytest.raises(LargePanelRefusedError, match="production test draw 1"):
        store.get_or_build(
            spec,
            make_panel(genes[:30], species="mouse"),
            builder=builder_for(spec, config),
            config=config,
        )
    assert not (tmp_path / "store").exists() or not any(
        (tmp_path / "store").rglob("bundle.json")
    )
    # The test set's spec of a prepared primary is the one its self-map builds.
    test_spec = reference.wmb_test_set_spec(spec, draw_config())
    assert test_spec.reference_id == reference.WMB_TESTSET_REFERENCE_ID
    assert set(test_spec.sources) == set(reference.WMB_TESTSET_SOURCES)


def test_no_production_build_hash_changes_and_only_the_test_set_takes_draw_2() -> None:
    """Regression pin: draw 1's and every seeded build_hash (the M3c golden)."""
    ctm = {"version": "1.7.2", "commit": None}
    golden_rows = pd.read_csv(
        Path(reference.__file__).parents[1]
        / "assets"
        / "annotation"
        / "validated_panels.csv"
    )
    mouse = golden_rows[golden_rows["species"] == "mouse"]
    assert set(mouse["panel_id"]) == {"mouse_ms1_500", "mouse_ms2_815"}
    for row in mouse.itertuples(index=False):
        panel = SimpleNamespace(
            panel_hash=str(row.panel_hash), n_genes=int(row.n_genes), species="mouse"
        )
        sources = {name: _fake_source(name) for name in ("a", "b")}
        hashes: dict[int, dict[str, str]] = {}
        for draw in (1, 2):
            config = with_draw(AnnotationConfig(species="mouse"), draw)
            hashes[draw] = {}
            for reference_id in ("wmb_panel", reference.WMB_TESTSET_REFERENCE_ID):
                spec = AnnotationReferenceSpec(
                    reference_id=reference_id, **KNOWN_REFERENCES[reference_id]
                )
                payload = build_hash_payload(
                    spec,
                    cast("Any", panel),
                    builder=builder_for(spec, config),
                    sources=sources,
                    config=config,
                    ctm=ctm,
                )
                hashes[draw][reference_id] = compute_build_hash(payload)
        golden = golden_for_alias_state()
        for reference_id, value in hashes[1].items():
            assert value == golden[f"build_hash/{row.panel_id}/{reference_id}"]
        assert (
            hashes[2][reference.WMB_TESTSET_REFERENCE_ID]
            != (hashes[1][reference.WMB_TESTSET_REFERENCE_ID])
        )


def test_the_reference_facts_give_the_composition_and_the_donors(
    tmp_path: Path,
) -> None:
    sources = write_wmb_draw_sources(tmp_path, n_per_cluster=25)
    spec = wmb_draw_spec(sources)
    facts = reference.wmb_reference_facts(
        spec.sources, ["cell-1-0", "cell-9-4", "missing"]
    )
    assert facts.n_cells == 9 * 25
    assert sum(facts.composition.values()) == pytest.approx(1.0)
    # Subclass 320 holds clusters 7-9: 75 of 225 cells.
    assert facts.composition["CS20230722_SUBC_320"] == pytest.approx(75 / 225)
    assert facts.donor_of == {"cell-1-0": "D1", "cell-9-4": "D2"}
    assert set(facts.composition) == set(REAL_SUBCLASS_CLUSTERS)


def _load_script() -> Any:
    import importlib.util

    script = (
        Path(__file__).resolve().parents[2] / "scripts/annotation/build_wmb_draw2.py"
    )
    spec = importlib.util.spec_from_file_location("build_wmb_draw2", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_build_script_draws_2_against_the_stored_draw_1(tmp_path: Path) -> None:
    sources = write_wmb_draw_sources(tmp_path)
    genes = json.loads(str(sources["genes"]))
    panel = make_panel(genes[:30], species="mouse")
    panel_file = panel.write(tmp_path / "panel_genes.json")
    config = draw_config()
    config = config.model_copy(
        update={
            "references": [
                AnnotationReferenceSpec(
                    reference_id="wmb_panel",
                    max_cells_per_cluster=20,
                    **{
                        key: value
                        for key, value in KNOWN_REFERENCES["wmb_panel"].items()
                        if key != "max_cells_per_cluster"
                    },
                )
            ]
        }
    )
    config_file = tmp_path / "annotation_config.json"
    config_file.write_text(config.model_dump_json())
    store_root = tmp_path / "store"
    arguments = [
        "--annotation-config",
        str(config_file),
        "--panel-genes",
        str(panel_file),
        "--store",
        str(store_root),
        "--source",
        f"wmb_h5ad_dir={sources['h5ad_dir']}",
        "--source",
        f"wmb_metadata_dir={sources['metadata']}",
        "--source",
        f"wmb_mapping_stats={sources['truth']}",
        "--source",
        f"wmb_selfmap_test_cells={sources['truth']}",
        "--output",
        str(tmp_path / "bundle_ref.json"),
        "--record",
        str(tmp_path / "record.json"),
    ]
    script = _load_script()
    # Without draw 1 in the store there is nothing to draw against.
    with pytest.raises(SystemExit, match="draw 1"):
        script.main(arguments)
    # Draw 1 as wmb_panel's self-map builds it (the spec derived from the
    # prepared primary spec, the path gate P takes too).
    primary = reference.prepare_reference_spec(
        config.references[0].model_copy(
            update={
                "sources": {
                    "wmb_h5ad_dir": sources["h5ad_dir"],
                    "wmb_metadata_dir": sources["metadata"],
                    "wmb_mapping_stats": sources["truth"],
                    "wmb_selfmap_test_cells": sources["truth"],
                }
            }
        ),
        reference.SourceOptions(download_dir=store_root / ".downloads"),
    )
    test_spec = reference.wmb_test_set_spec(primary, config)
    first_dir, _ = build(ReferenceStore(store_root), test_spec, panel, config)
    with pytest.raises(SystemExit, match="is not"):
        script.main([*arguments, "--expect-draw1", "ffff"])
    assert script.main([*arguments, "--expect-draw1", first_dir.name[:12]]) == 0
    record = json.loads((tmp_path / "record.json").read_text())
    assert all(record["checks"].values())
    assert record["draw1"]["build_hash"] == first_dir.name
    second = Path(record["draw2"]["path"])
    assert second.parent == first_dir.parent and second != first_dir
    assert record["gate_p_draw"]["n_cells"] == len(res.load_test_cells(second).obs)
    # A re-run reuses it.
    assert script.main(arguments) == 0
    assert json.loads((tmp_path / "record.json").read_text())["draw2"]["reused"]
