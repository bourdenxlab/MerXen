"""Tests for the MapMyCells engine of the MAP step (plan §3.3, D-A8)."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from merxen.annotation import mapmycells_engine as engine
from merxen.annotation.mapmycells_engine import (
    MMC_ENTRYPOINT_MODULE,
    MMC_SINGLE_THREAD_ENV,
    TIDY_COLUMNS,
    CtmVersionError,
    ExtendedJsonError,
    MmcBundle,
    MmcEngineError,
    MmcEngineParams,
    aggregate_parent_probability,
    build_mmc_command,
    check_ctm_version,
    level_frame,
    mmc_environment,
    parse_extended_json_tidy,
    query_fingerprint,
    read_tidy_parquet,
    restrict_lookup,
    run_mmc,
    write_query_h5ad,
)

from .conftest import FakeMmc, FakeNode

SUPC, CLUS = "FAKE_SUPC", "FAKE_CLUS"
NODES = [
    FakeNode("N1", "Exc", "ENSG00000000001", {"broad_class": "Neurons"}),
    FakeNode("N2", "Inh", "ENSG00000000002", {"broad_class": "Neurons"}),
    FakeNode("N3", "Astro", "ENSG00000000003", {"broad_class": "Astrocytes"}),
]


def _extended_payload() -> dict[str, Any]:
    """A fixture extended JSON: 2 cells x 2 levels, as ctm 1.7.2 writes it."""
    return {
        "taxonomy_tree": {
            "hierarchy": [SUPC, CLUS],
            "hierarchy_mapper": {SUPC: "supercluster", CLUS: "cluster"},
            "name_mapper": {
                SUPC: {"N1": {"name": "Exc"}, "N2": {"name": "Inh"}},
                CLUS: {"C1": {"name": "Exc L2/3"}},
            },
        },
        "results": [
            {
                SUPC: {
                    "assignment": "N1",
                    "bootstrapping_probability": 0.7,
                    "aggregate_probability": 0.7,
                    "avg_correlation": 0.42,
                    "runner_up_assignment": ["N2", "N3"],
                    "runner_up_correlation": [0.3, 0.2],
                    "runner_up_probability": [0.2, 0.1],
                    "directly_assigned": True,
                },
                CLUS: {
                    "assignment": "C1",
                    "bootstrapping_probability": 0.9,
                    "aggregate_probability": 0.63,
                    "avg_correlation": 0.4,
                    "runner_up_assignment": [],
                    "runner_up_correlation": [],
                    "runner_up_probability": [],
                    "directly_assigned": False,
                },
                "cell_id": "b",
            },
            {
                SUPC: {
                    "assignment": "N3",
                    "bootstrapping_probability": 1.0,
                    "aggregate_probability": 1.0,
                    "avg_correlation": 0.8,
                    "runner_up_assignment": ["N1", "N2", "N4", "N5", "N6", "N7"],
                    "runner_up_correlation": [0.1] * 6,
                    "runner_up_probability": [0.0] * 6,
                    "directly_assigned": True,
                },
                CLUS: {
                    "assignment": "C3",
                    "bootstrapping_probability": 1.0,
                    "aggregate_probability": 1.0,
                    "avg_correlation": 0.8,
                    "directly_assigned": True,
                },
                "cell_id": "a",
            },
        ],
        "config": {
            "type_assignment": {
                "rng_seed": 0,
                "bootstrap_factor": 0.5,
                "bootstrap_iteration": 100,
                "n_processors": 6,
                "n_runners_up": 5,
                "normalization": "raw",
            }
        },
        "metadata": {"version": "1.7.2"},
    }


def test_parse_extended_json_tidy_reads_every_field(tmp_path: Path) -> None:
    path = tmp_path / "extended.json"
    path.write_text(json.dumps(_extended_payload()))

    tidy = parse_extended_json_tidy(path, names={SUPC: {"N3": "Astro (tree)"}})

    assert list(tidy.columns) == list(TIDY_COLUMNS)
    assert len(tidy) == 4
    supc = level_frame(tidy, SUPC)
    assert list(supc.index) == ["b", "a"]
    first = supc.loc["b"]
    assert first["level_name"] == "supercluster"
    assert first["assignment"] == "N1"
    assert first["name"] == "Exc"
    assert first["bp"] == pytest.approx(0.7)
    assert first["aggregate_probability"] == pytest.approx(0.7)
    assert first["avg_correlation"] == pytest.approx(0.42)
    assert bool(first["directly_assigned"]) is True
    assert first["n_runners_up"] == 2
    assert first["runner_up_1_assignment"] == "N2"
    assert first["runner_up_1_name"] == "Inh"
    assert first["runner_up_1_probability"] == pytest.approx(0.2)
    assert first["runner_up_2_correlation"] == pytest.approx(0.2)
    assert pd.isna(first["runner_up_3_assignment"])
    assert np.isnan(first["runner_up_3_probability"])
    # Names missing from the JSON come from the bundle tree, else the label.
    assert supc.loc["a", "name"] == "Astro (tree)"
    assert supc.loc["a", "n_runners_up"] == 5
    assert supc.loc["a", "runner_up_5_assignment"] == "N6"
    cluster = level_frame(tidy, "cluster")
    assert cluster.loc["b", "name"] == "Exc L2/3"
    assert bool(cluster.loc["b", "directly_assigned"]) is False
    assert cluster.loc["a", "name"] == "C3"
    assert tidy["bp"].dtype == np.float32
    assert isinstance(tidy["assignment"].dtype, pd.CategoricalDtype)
    assert str(tidy["directly_assigned"].dtype) == "boolean"


def test_parse_extended_json_follows_the_query_order() -> None:
    tidy = parse_extended_json_tidy(_extended_payload(), cell_order=["a", "b"])

    assert list(level_frame(tidy, SUPC).index) == ["a", "b"]


def test_parse_extended_json_rejects_cells_that_do_not_match_the_query() -> None:
    with pytest.raises(ExtendedJsonError, match="do not match the query"):
        parse_extended_json_tidy(_extended_payload(), cell_order=["a", "c"])
    with pytest.raises(ExtendedJsonError, match="do not match the query"):
        parse_extended_json_tidy(_extended_payload(), cell_order=["a"])


def test_parse_extended_json_rejects_a_missing_level_entry() -> None:
    payload = _extended_payload()
    del payload["results"][1][CLUS]

    with pytest.raises(ExtendedJsonError, match="no FAKE_CLUS entry"):
        parse_extended_json_tidy(payload)


def test_parse_extended_json_rejects_repeated_cells() -> None:
    payload = _extended_payload()
    payload["results"][1]["cell_id"] = "b"

    with pytest.raises(ExtendedJsonError, match="repeats cell ids"):
        parse_extended_json_tidy(payload)


def test_query_fingerprint_is_stable() -> None:
    first = query_fingerprint(["c1", "c2"], np.array([10, 25]), ["G1", "G2"])

    assert first == query_fingerprint(["c1", "c2"], [10.0, 25.0], ["G1", "G2"])
    assert first == query_fingerprint(
        pd.Index(["c1", "c2"]), np.array([10, 25], dtype=np.int32), ("G1", "G2")
    )
    assert len(first) == 64


@pytest.mark.parametrize(
    ("obs", "counts", "genes"),
    [
        (["c1", "c3"], [10, 25], ["G1", "G2"]),
        (["c2", "c1"], [25, 10], ["G1", "G2"]),
        (["c1", "c2"], [10, 26], ["G1", "G2"]),
        (["c1", "c2"], [10, 25], ["G1", "G3"]),
        (["c1", "c2"], [10, 25], ["G2", "G1"]),
        (["c1", "c2"], [10, 25], ["G1"]),
        (["c1c", "2"], [10, 25], ["G1", "G2"]),
    ],
)
def test_query_fingerprint_changes_with_any_query_change(
    obs: list[str], counts: list[int], genes: list[str]
) -> None:
    base = query_fingerprint(["c1", "c2"], [10, 25], ["G1", "G2"])

    assert query_fingerprint(obs, counts, genes) != base


def test_query_fingerprint_needs_one_count_per_cell() -> None:
    with pytest.raises(ValueError, match="total counts"):
        query_fingerprint(["c1"], [1, 2], ["G1"])


def test_check_ctm_version_accepts_the_configured_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine.importlib.metadata, "version", lambda name: "1.7.2")

    assert check_ctm_version("1.7.2") == "1.7.2"


def test_check_ctm_version_refuses_another_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine.importlib.metadata, "version", lambda name: "1.5.5")

    with pytest.raises(CtmVersionError, match=r"1\.5\.5 .*needs 1\.7\.2"):
        check_ctm_version("1.7.2")


def test_check_ctm_version_refuses_a_missing_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str) -> str:
        raise engine.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(engine.importlib.metadata, "version", missing)

    with pytest.raises(CtmVersionError, match="is not installed"):
        check_ctm_version("1.7.2")


def test_mmc_environment_pins_one_blas_thread_and_hides_gpus() -> None:
    environment = mmc_environment(
        {"OMP_NUM_THREADS": "16", "CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": "/x"}
    )

    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        assert environment[name] == "1"
    assert set(MMC_SINGLE_THREAD_ENV) <= set(environment)
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    source = str(Path(engine.__file__).resolve().parents[2])
    assert environment["PYTHONPATH"].split(":") == [source, "/x"]


def test_engine_params_default_to_the_validated_configuration() -> None:
    from merxen.annotation.config import default_references

    spec = default_references("human")[0]
    params = MmcEngineParams.from_reference_spec(spec, n_processors=6)

    assert params.bootstrap_factor == 0.5
    assert params.bootstrap_iteration == 100
    assert params.rng_seed == 0
    assert params.n_processors == 6
    assert params.normalization == "raw"
    assert params.cloud_safe is False
    assert params.n_runners_up == 5


def _bundle(fake_mmc: FakeMmc, **kwargs: Any) -> MmcBundle:
    defaults: dict[str, Any] = {
        "role": "primary",
        "species": "human",
        "panel_hash": "p" * 64,
        "n_genes": 4,
        "levels": [SUPC, CLUS],
        "nodes": NODES,
    }
    defaults.update(kwargs)
    return MmcBundle.from_dir(fake_mmc.bundle("fake_ref", **defaults))


def test_build_mmc_command_uses_the_entrypoint_and_validated_settings(
    fake_mmc: FakeMmc, tmp_path: Path
) -> None:
    bundle = _bundle(fake_mmc)
    command = build_mmc_command(
        tmp_path / "q.h5ad",
        bundle,
        MmcEngineParams(n_processors=6),
        extended_json=tmp_path / "e.json",
        log_path=tmp_path / "log.txt",
        tmp_dir=tmp_path / "tmp",
    )

    assert command[:3] == [sys.executable, "-m", MMC_ENTRYPOINT_MODULE]

    def value(name: str) -> str:
        return command[command.index(name) + 1]

    assert value("--type_assignment.bootstrap_factor") == "0.5"
    assert value("--type_assignment.bootstrap_iteration") == "100"
    assert value("--type_assignment.rng_seed") == "0"
    assert value("--type_assignment.n_processors") == "6"
    assert value("--type_assignment.normalization") == "raw"
    assert value("--type_assignment.n_runners_up") == "5"
    assert value("--cloud_safe") == "False"
    assert value("--precomputed_stats.path") == str(bundle.mapping_precompute)
    assert value("--query_markers.serialized_lookup") == str(bundle.lookup)
    assert "--drop_level" not in command
    assert "--nodes_to_drop" not in command


def test_build_mmc_command_drops_the_bundle_level_and_nodes(
    fake_mmc: FakeMmc, tmp_path: Path
) -> None:
    bundle = _bundle(
        fake_mmc,
        species="mouse",
        levels=["W_CLAS", "W_SUBC", "W_SUPT", "W_CLUS"],
        drop_level="W_SUPT",
    )
    command = build_mmc_command(
        tmp_path / "q.h5ad",
        bundle,
        MmcEngineParams(n_processors=6),
        extended_json=tmp_path / "e.json",
        log_path=tmp_path / "log.txt",
        lookup_path=tmp_path / "restricted.json",
        nodes_to_drop=[("W_SUBC", "N2_1")],
    )

    assert bundle.levels == ("W_CLAS", "W_SUBC", "W_CLUS")
    assert command[command.index("--drop_level") + 1] == "W_SUPT"
    assert json.loads(command[command.index("--nodes_to_drop") + 1]) == [
        ["W_SUBC", "N2_1"]
    ]
    assert command[command.index("--query_markers.serialized_lookup") + 1] == str(
        tmp_path / "restricted.json"
    )


def _query(tmp_path: Path, genes: list[str] | None = None) -> Path:
    genes = genes or [node.marker for node in NODES] + ["ENSG00000000009"]
    counts = np.zeros((3, len(genes)), dtype=np.int64)
    counts[0, 0] = 9
    counts[0, 1] = 1
    counts[1, 2] = 12
    counts[2, 1] = 4
    counts[2, 0] = 4
    return write_query_h5ad(counts, ["c1", "c2", "c3"], genes, tmp_path / "q.h5ad")


def test_run_mmc_writes_the_tidy_parquet_and_deletes_the_json(
    fake_mmc: FakeMmc, tmp_path: Path
) -> None:
    bundle = _bundle(fake_mmc)
    work = tmp_path / "work"

    result = run_mmc(
        _query(tmp_path),
        bundle,
        MmcEngineParams(n_processors=6),
        output_parquet=tmp_path / "out" / "S_mmc_fake_ref.parquet",
        work_dir=work,
        expected_ctm_version="1.7.2",
        run_metadata={"run_id": "fake_ref"},
    )

    tidy, metadata = read_tidy_parquet(result.parquet)
    assert result.n_cells == 3
    assert result.n_query_genes == 4
    assert result.levels == (SUPC, CLUS)
    supc = level_frame(tidy, SUPC)
    assert list(supc.index) == ["c1", "c2", "c3"]
    assert list(supc["assignment"]) == ["N1", "N3", "N1"]
    assert supc.loc["c1", "bp"] == pytest.approx(0.9)
    assert supc.loc["c1", "runner_up_1_assignment"] == "N2"
    assert metadata["build_hash"] == bundle.build_hash
    assert metadata["run_id"] == "fake_ref"
    assert metadata["engine_params"]["rng_seed"] == 0
    assert result.effective_config["type_assignment"]["n_processors"] == 6
    assert result.extended_json is None
    assert not list(work.glob("*.json"))
    assert not list(work.glob("*ctm_tmp"))
    assert result.stdout_log.read_text().startswith("$ ")
    call = fake_mmc.calls[-1]
    for name in MMC_SINGLE_THREAD_ENV:
        assert call["env"][name] == "1"
    assert call["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert MMC_ENTRYPOINT_MODULE in call["command"]


def test_run_mmc_keeps_the_extended_json_gzipped_on_request(
    fake_mmc: FakeMmc, tmp_path: Path
) -> None:
    result = run_mmc(
        _query(tmp_path),
        _bundle(fake_mmc),
        MmcEngineParams(n_processors=2),
        output_parquet=tmp_path / "out" / "S_mmc_fake_ref.parquet",
        work_dir=tmp_path / "work",
        expected_ctm_version="1.7.2",
        keep_extended_json=True,
    )

    assert result.extended_json is not None
    assert result.extended_json.name == "S_mmc_fake_ref.extended.json.gz"
    with gzip.open(result.extended_json, "rt") as handle:
        assert len(json.load(handle)["results"]) == 3
    assert not list((tmp_path / "work").glob("*.json"))


def test_run_mmc_refuses_another_ctm_version_before_mapping(
    fake_mmc: FakeMmc, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(engine, "installed_ctm_version", lambda: "1.5.5")

    with pytest.raises(CtmVersionError):
        run_mmc(
            _query(tmp_path),
            _bundle(fake_mmc),
            MmcEngineParams(),
            output_parquet=tmp_path / "o.parquet",
            work_dir=tmp_path / "work",
            expected_ctm_version="1.7.2",
        )
    assert fake_mmc.calls == []


def test_run_mmc_reports_a_failed_mapper(fake_mmc: FakeMmc, tmp_path: Path) -> None:
    fake_mmc.fail_with = 3

    with pytest.raises(MmcEngineError, match=r"(?s)exit code 3.*fake mapper failed"):
        run_mmc(
            _query(tmp_path),
            _bundle(fake_mmc),
            MmcEngineParams(),
            output_parquet=tmp_path / "o.parquet",
            work_dir=tmp_path / "work",
            expected_ctm_version="1.7.2",
        )
    assert not (tmp_path / "o.parquet").exists()


def test_run_mmc_refuses_a_mapping_with_other_recorded_settings(
    fake_mmc: FakeMmc, tmp_path: Path
) -> None:
    fake_mmc.record_seed = 11235813

    with pytest.raises(MmcEngineError, match="rng_seed"):
        run_mmc(
            _query(tmp_path),
            _bundle(fake_mmc),
            MmcEngineParams(),
            output_parquet=tmp_path / "o.parquet",
            work_dir=tmp_path / "work",
            expected_ctm_version="1.7.2",
        )


def test_run_mmc_refuses_a_mapping_that_recorded_another_drop_level(
    fake_mmc: FakeMmc, tmp_path: Path
) -> None:
    fake_mmc.record_drop_level = "X"

    with pytest.raises(MmcEngineError, match="drop_level"):
        run_mmc(
            _query(tmp_path),
            _bundle(fake_mmc),
            MmcEngineParams(),
            output_parquet=tmp_path / "o.parquet",
            work_dir=tmp_path / "work",
            expected_ctm_version="1.7.2",
        )
    assert not (tmp_path / "o.parquet").exists()


def test_the_fake_mapper_leaves_subprocess_run_alone(
    fake_mmc: FakeMmc, tmp_path: Path
) -> None:
    import subprocess
    import sys

    completed = subprocess.run(
        [sys.executable, "-c", "print('real')"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "real"
    assert fake_mmc.calls == []


def test_mmc_bundle_from_dir_refuses_incomplete_or_changed_bundles(
    fake_mmc: FakeMmc,
) -> None:
    bundle_dir = fake_mmc.bundle(
        "broken",
        role="primary",
        species="human",
        panel_hash="p" * 64,
        n_genes=4,
        levels=[SUPC, CLUS],
        nodes=NODES,
    )
    lookup = bundle_dir / "query_markers.filtered.json"
    lookup.write_text(json.dumps({"None": ["ENSG00000000001", "ENSG00000000009"]}))

    with pytest.raises(MmcEngineError, match="size of|does not match the digest"):
        MmcBundle.from_dir(bundle_dir)

    (bundle_dir / "mapping_precompute.h5").unlink()
    with pytest.raises(MmcEngineError, match="missing 'mapping_precompute.h5'"):
        MmcBundle.from_dir(bundle_dir)

    manifest = json.loads((bundle_dir / "bundle.json").read_text())
    manifest["status"] = "building"
    (bundle_dir / "bundle.json").write_text(json.dumps(manifest))
    with pytest.raises(MmcEngineError, match="not complete"):
        MmcBundle.from_dir(bundle_dir)


def test_restrict_lookup_only_when_a_marker_is_missing(
    fake_mmc: FakeMmc, tmp_path: Path
) -> None:
    bundle = _bundle(fake_mmc)
    genes = [node.marker for node in NODES]

    unchanged = restrict_lookup(
        bundle, genes + ["ENSG00000000009"], tmp_path / "r.json"
    )
    assert unchanged.path is None
    assert unchanged.lookup_sha256 == bundle.lookup_sha256

    restricted = restrict_lookup(bundle, genes[1:], tmp_path / "r.json")
    assert restricted.path == tmp_path / "r.json"
    written = json.loads(restricted.path.read_text())
    assert written["None"] == genes[1:]
    assert restricted.lookup_sha256 != bundle.lookup_sha256
    assert restricted.validation is not None


def test_aggregate_parent_probability_sums_runner_ups_of_the_same_class() -> None:
    tidy = parse_extended_json_tidy(_extended_payload(), cell_order=["b", "a"])
    frame = level_frame(tidy, SUPC)

    result = aggregate_parent_probability(
        frame, {"N1": "Neurons", "N2": "Neurons", "N3": "Astrocytes"}
    )

    assert list(result.classes) == ["Neurons", "Astrocytes"]
    np.testing.assert_allclose(result.probability, [0.9, 1.0], rtol=1e-6)
    assert list(result.runner_up_class) == ["Astrocytes", "Neurons"]
    np.testing.assert_allclose(result.runner_up_probability, [0.1, 0.0], atol=1e-6)


def test_aggregate_parent_probability_gives_nan_without_a_class() -> None:
    frame = level_frame(parse_extended_json_tidy(_extended_payload()), SUPC)

    result = aggregate_parent_probability(frame, {"N3": "Astrocytes"})

    assert result.classes[0] is None
    assert np.isnan(result.probability[0])


def test_write_query_h5ad_round_trips_counts_and_ids(tmp_path: Path) -> None:
    import anndata as ad

    counts = sparse.csr_matrix(np.array([[1, 0], [0, 3]], dtype=np.int64))
    path = write_query_h5ad(counts, ["x", "y"], ["G1", "G2"], tmp_path / "q.h5ad")

    adata = ad.read_h5ad(path)
    assert list(adata.obs_names) == ["x", "y"]
    assert list(adata.var_names) == ["G1", "G2"]
    np.testing.assert_array_equal(adata.X.toarray(), counts.toarray())
    with pytest.raises(ValueError, match="unique"):
        write_query_h5ad(counts, ["x", "x"], ["G1", "G2"], tmp_path / "bad.h5ad")
