"""The Groovy samplesheet reader and config emitter of manual registration.

``workflows/lib/ManualRegistrationSheet.groovy`` is pure, so one Nextflow launch
runs it on every case below (copy of the library + a harness class in a temporary
project). The Python side builds the sheets, the placeholder images and the fake
prepared stores, then reads one JSON result file. Every test function reads that
shared result. The launch runs once per session (also under pytest-xdist).
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from merxen.config import ViewerCacheConfig
from merxen.manual_registration.config import (
    ManualRegistrationCompareConfig,
    ManualRegistrationPrepareConfig,
    ManualRegistrationRunConfig,
    ManualRegistrationSettings,
)
from merxen.manual_registration.presets import FORBIDDEN_OVERRIDES

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / "workflows"
NEXTFLOW = shutil.which("nextflow")
needs_nextflow = pytest.mark.skipif(NEXTFLOW is None, reason="nextflow is unavailable")
pytestmark = [needs_nextflow, pytest.mark.nextflow]

STORE_ATTR = "merxen_manual_registration_store"
LABELLING_ATTR = "merxen_manual_registration"

HARNESS_CLASS = """
import groovy.json.JsonOutput
import groovy.json.JsonSlurper
import java.nio.file.Paths

class SheetHarness {
    static void run(Object casesPath, Object outPath) {
        Map cases = new JsonSlurper().parse(new File(casesPath.toString())) as Map
        Map results = [parse: [:], store: [:], min_pairs: [:], names: [:]]
        def S = ManualRegistrationSheet
        (cases.parse as Map).each { name, raw ->
            Map c = raw as Map
            Map p = c.params as Map
            String mode = c.mode as String
            Map parsed = S.parse(Paths.get(c.sheet as String), p, mode)
            Map r = [
                errors: parsed.errors, warnings: parsed.warnings, rows: parsed.rows,
            ]
            if (!parsed.errors) {
                r.preflight = mode == "register" ?
                    S.registerPreflight(parsed.rows, p) : []
                r.prepare = parsed.rows.collect { row -> S.prepareConfigJson(row, p) }
                r.viewer = parsed.rows.collect { row ->
                    S.viewerCacheConfigJson(row, p)
                }
                r.runs = S.plannedRuns(parsed.rows, p)
                r.compare = S.compareConfigJson("X1", ["/a/s1.json", "/a/s2.json"], p)
            }
            r.czi_python = S.cziPython(p)
            results.parse[name] = r
        }
        (cases.store as Map).each { name, path ->
            results.store[name] = S.storeState(path as String)
        }
        results.settings_keys = S.SETTINGS_KEYS
        results.allowed_columns = S.ALLOWED_COLUMNS
        results.names = [
            run_name: S.runName("serial_section", "trial"),
            run_name_blank: S.runName("same_section", null),
            run_id: S.runId("LCM1", "serial_section", null),
            prepared_dir_relative: S.preparedDir([outdir: "rel/out"], "LCM1"),
        ]
        new File(outPath.toString()).text = JsonOutput.toJson(results)
    }
}
"""
HARNESS_MAIN = "workflow {\n    SheetHarness.run(params.cases, params.out)\n}\n"

JVM_OPTIONS = (
    f"-XX:TieredStopAtLevel=1 -XX:ActiveProcessorCount={min(4, os.cpu_count() or 1)}"
)

HEADER = ["id", "reference_image", "moving_image"]


# --- helpers ---------------------------------------------------------------


def _csv_field(value: str) -> str:
    if any(char in value for char in ',"\n'):
        return '"' + value.replace('"', '""') + '"'
    return value


def _write_sheet(
    path: Path, header: list[str], rows: list[list[str]], *, newline: str = "\n"
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        ",".join(_csv_field(cell) for cell in record) for record in [header, *rows]
    ]
    path.write_bytes((newline.join(lines) + newline).encode("utf-8"))
    return path


def _fake_store(
    path: Path,
    *,
    kind: str | None = "prepared",
    dataset_name: str | None = "DS",
    status: str | None = "finished",
    n_enabled: int = 8,
    n_disabled: int = 0,
    store_version: int = 1,
    fmt: str = "v3",
    node_type: str = "group",
    drop_enabled_key: bool = False,
) -> Path:
    """A directory with the root attributes of a store (no arrays).

    ``fmt``: ``v3`` (zarr.json), ``v2`` (.zgroup + .zattrs) or ``v2_no_zgroup`` (a
    .zattrs without the .zgroup that makes the directory a group).
    """
    path.mkdir(parents=True, exist_ok=True)
    attrs: dict[str, Any] = {}
    if kind is not None:
        attrs[STORE_ATTR] = {
            "schema": "merxen.manual_registration.store",
            "schema_version": store_version,
            "kind": kind,
        }
        if dataset_name is not None:
            attrs[STORE_ATTR]["dataset_name"] = dataset_name
    if status is not None:
        pairs: list[dict[str, Any]] = []
        for index in range(n_enabled + n_disabled):
            pair: dict[str, Any] = {"id": index + 1, "enabled": index < n_enabled}
            if drop_enabled_key and pair["enabled"]:
                del pair["enabled"]
            pairs.append(pair)
        attrs[LABELLING_ATTR] = {
            "schema_version": 1,
            "status": status,
            "landmarks": {"pairs": pairs},
        }
    if fmt == "v3":
        document = {"zarr_format": 3, "node_type": node_type, "attributes": attrs}
        (path / "zarr.json").write_text(json.dumps(document))
    else:
        (path / ".zarray_placeholder").write_text("")
        if fmt == "v2":
            (path / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
        (path / ".zattrs").write_text(json.dumps(attrs))
    return path


class _Project:
    """The directories of one session launch."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.project = root / "project"
        self.fs = root / "fs"
        self.img = self.fs / "img"
        self.sheets = self.fs / "sheets"
        self.stores = self.fs / "stores"
        self.cases: dict[str, dict[str, Any]] = {}
        self.stores_cases: dict[str, str] = {}
        for folder in (self.project / "lib", self.img, self.sheets, self.stores):
            folder.mkdir(parents=True, exist_ok=True)
        for name in ("a.czi", "b.czi", "c.ome.tif", "d.ome.tiff", "e.tif", "f.png"):
            (self.img / name).write_bytes(b"x")
        (self.img / "with,comma").mkdir()
        (self.img / "with,comma" / "g.ome.tif").write_bytes(b"x")
        (self.img / "link.czi").symlink_to(self.img / "a.czi")

    @property
    def params(self) -> dict[str, Any]:
        return {
            "outdir": str(self.fs / "out"),
            "manual_registration_presets": "same_section",
            "manual_registration_settings": "",
            "manual_registration_force_prepare": False,
            "manual_registration_force_register": False,
            "manual_registration_czi_python": "in_process",
            "viewer_cache_downsample": 4,
            "viewer_cache_min_size": 1024,
        }

    def add(
        self,
        name: str,
        header: list[str],
        rows: list[list[str]],
        *,
        mode: str = "prepare",
        params: dict[str, Any] | None = None,
        sheet_dir: Path | None = None,
        newline: str = "\n",
        text: str | None = None,
    ) -> None:
        folder = sheet_dir or self.sheets
        sheet = folder / f"{name}.csv"
        if text is None:
            _write_sheet(sheet, header, rows, newline=newline)
        else:
            sheet.parent.mkdir(parents=True, exist_ok=True)
            sheet.write_bytes(text.encode("utf-8"))
        merged = {**self.params, **(params or {})}
        self.cases[name] = {"sheet": str(sheet), "mode": mode, "params": merged}

    def ref(self, name: str = "a.czi") -> str:
        return str(self.img / name)


def _build_cases(p: _Project) -> None:
    img = p.ref
    p.add("minimal", HEADER, [["X1", img("a.czi"), img("c.ome.tif")]])

    p.add(
        "relative_paths",
        HEADER,
        [["X1", "../img/a.czi", "../img/c.ome.tif"]],
    )

    quoted = (
        "﻿id,reference_image,moving_image,moving_channel_names\r\n"
        f'X1,{img("a.czi")},"{p.img / "with,comma" / "g.ome.tif"}","DAPI;AF568"\r\n'
    )
    p.add("quoted_bom_crlf", [], [], text=quoted)

    p.add("missing_column", ["id", "reference_image"], [["X1", img("a.czi")]])
    p.add(
        "unknown_column",
        [*HEADER, "foo"],
        [["X1", img("a.czi"), img("c.ome.tif"), "1"]],
    )
    p.add(
        "duplicate_header",
        ["id", "id", "reference_image", "moving_image"],
        [["X1", "X1", img("a.czi"), img("c.ome.tif")]],
    )
    p.add(
        "bad_and_duplicate_ids",
        HEADER,
        [
            ["9-ok", img("a.czi"), img("c.ome.tif")],
            ["bad id", img("a.czi"), img("c.ome.tif")],
            ["9-ok", img("a.czi"), img("c.ome.tif")],
        ],
    )
    p.add(
        "extensions_and_missing",
        HEADER,
        [
            ["X1", img("f.png"), img("c.ome.tif")],
            ["X2", img("a.czi"), str(p.img / "missing.czi")],
        ],
    )
    p.add(
        "same_file",
        HEADER,
        [
            ["X1", img("a.czi"), img("link.czi")],
            ["X2", img("c.ome.tif"), img("c.ome.tif")],
        ],
    )
    p.add(
        "scene_on_tiff",
        [*HEADER, "reference_scene"],
        [["X1", img("c.ome.tif"), img("a.czi"), "S0"]],
    )
    p.add(
        "scene_on_czi",
        [*HEADER, "reference_scene"],
        [["X1", img("a.czi"), img("c.ome.tif"), "ScanRegion0"]],
    )
    p.add(
        "channel_names",
        [*HEADER, "reference_channel_names", "moving_channel_names"],
        [
            ["X1", img("a.czi"), img("c.ome.tif"), "DAPI;;AF568", ""],
            ["X2", img("a.czi"), img("c.ome.tif"), "", "DAPI;dapi"],
        ],
    )
    p.add(
        "pixel_size",
        [*HEADER, "reference_pixel_size_um", "moving_pixel_size_um"],
        [
            ["X1", img("a.czi"), img("c.ome.tif"), "0", ""],
            ["X2", img("a.czi"), img("c.ome.tif"), "-1", ""],
            ["X3", img("a.czi"), img("c.ome.tif"), "abc", ""],
        ],
    )
    p.add(
        "pixel_size_ok",
        [*HEADER, "reference_pixel_size_um", "moving_pixel_size_um"],
        [["X4", img("a.czi"), img("c.ome.tif"), "", "0.34511"]],
    )
    p.add(
        "presets_bad",
        [*HEADER, "presets"],
        [
            ["X1", img("a.czi"), img("c.ome.tif"), "fast"],
            ["X2", img("a.czi"), img("c.ome.tif"), "same_section;same_section"],
        ],
    )
    p.add(
        "presets_inherit",
        [*HEADER, "presets"],
        [["X1", img("a.czi"), img("c.ome.tif"), ""]],
        params={"manual_registration_presets": "same_section;serial_section"},
    )
    for index, (item, _expected) in enumerate(SETTINGS_CASES):
        p.add(
            f"settings_{index}",
            [*HEADER, "registration_settings"],
            [["X1", img("a.czi"), img("c.ome.tif"), item]],
        )
    p.add(
        "params_checks",
        HEADER,
        [["X1", img("a.czi"), img("c.ome.tif")]],
        params={
            "manual_registration_presets": "fast",
            "manual_registration_force_prepare": "maybe",
            "manual_registration_settings": "serial_section:Bad=1",
        },
    )
    p.add(
        "prepare_mode_prepared_store",
        [*HEADER, "prepared_store"],
        [["X1", img("a.czi"), img("c.ome.tif"), "/elsewhere/X.zarr"]],
    )

    # CZI runtime param (prepare reads CZI; register never does).
    czi_row = [["X1", img("a.czi"), img("c.ome.tif")]]
    tiff_row = [["X1", img("c.ome.tif"), img("d.ome.tiff")]]
    fake_python = p.fs / "fake_python"
    fake_python.write_text("#!/bin/sh\n")
    p.add("czi_unset", HEADER, czi_row, params={"manual_registration_czi_python": None})
    p.add(
        "czi_unset_both",
        HEADER,
        [["X1", img("a.czi"), img("b.czi")]],
        params={"manual_registration_czi_python": None},
    )
    p.add(
        "czi_relative",
        HEADER,
        czi_row,
        params={"manual_registration_czi_python": "rel/python"},
    )
    p.add(
        "czi_missing",
        HEADER,
        czi_row,
        params={"manual_registration_czi_python": str(p.fs / "no_such_python")},
    )
    p.add(
        "czi_quote",
        HEADER,
        czi_row,
        params={"manual_registration_czi_python": str(p.fs / "it's")},
    )
    p.add(
        "czi_existing",
        HEADER,
        czi_row,
        params={"manual_registration_czi_python": str(fake_python)},
    )
    p.add(
        "czi_in_process",
        HEADER,
        czi_row,
        params={"manual_registration_czi_python": "in_process"},
    )
    p.add(
        "czi_register_mode",
        HEADER,
        czi_row,
        mode="register",
        params={"manual_registration_czi_python": None},
    )
    p.add(
        "czi_tiff_only",
        HEADER,
        tiff_row,
        params={"manual_registration_czi_python": None},
    )

    # The CZI check covers every row, also one with other errors.
    p.add(
        "czi_unset_and_row_errors",
        [*HEADER, "reference_pixel_size_um"],
        [
            ["X1", img("a.czi"), img("c.ome.tif"), "abc"],
            ["X2", img("c.ome.tif"), img("d.ome.tiff"), "abc"],
            ["bad id", img("a.czi"), img("b.czi"), ""],
            ["X1", img("a.czi"), img("c.ome.tif"), ""],
        ],
        params={"manual_registration_czi_python": None},
    )

    # Row numbers are the physical record numbers: blank lines count.
    p.add(
        "blank_lines",
        [],
        [],
        text=(
            "id,reference_image,moving_image\n\n"
            f"X1,{img('f.png')},{img('c.ome.tif')}\n\n\n"
            f"X2,{img('a.czi')},{img('c.ome.tif')}\n"
            f"X1,{img('a.czi')},{img('c.ome.tif')}\n"
        ),
    )
    p.add(
        "blank_line_then_open_quote",
        [],
        [],
        text='id,reference_image,moving_image\n\nX1,"abc\n',
    )

    # --outdir must not be the main pipeline's default results folder.
    one_row = [["X1", img("a.czi"), img("c.ome.tif")]]
    for name, value in [
        ("outdir_default", "./results"),
        ("outdir_bare", "results"),
        ("outdir_slash", "./results/"),
        ("outdir_dots", "./sub/../results"),
        ("outdir_blank", ""),
        ("outdir_other_relative", "./results_manual"),
        ("outdir_absolute_results", str(p.fs / "results")),
    ]:
        p.add(name, HEADER, one_row, params={"outdir": value})

    # Register preflight over fake stores (one store per row: a store belongs to one
    # dataset, and the dataset name must equal the row id).
    stores = p.stores
    preflight_rows: list[tuple[str, str, dict[str, Any] | None, str, str, str]] = [
        # id, store folder, store kwargs (None: the folder is not created),
        # registration_settings, run_label, presets
        ("P_MISSING", "missing.zarr", None, "", "", ""),
        ("P_V2", "v2.zarr", {"fmt": "v2"}, "", "", ""),
        ("P_DRAFT", "draft.zarr", {"status": "draft"}, "", "", ""),
        ("P_RESULT", "result.zarr", {"kind": "result"}, "", "", ""),
        ("P_NEWER", "newer.zarr", {"store_version": 2}, "", "", ""),
        # 8 pairs, one disabled: 7 enabled.
        ("P_SEVEN", "seven.zarr", {"n_enabled": 7, "n_disabled": 1}, "", "", ""),
        (
            "P_SIX",
            "six.zarr",
            {"n_enabled": 6, "n_disabled": 2},
            "landmark_min_pairs=6",
            "",
            "",
        ),
        (
            "P_SIX_PRESET",
            "six_preset.zarr",
            {"n_enabled": 6, "n_disabled": 2},
            "serial_section:landmark_min_pairs=7",
            "",
            "same_section;serial_section",
        ),
        ("A", "ok_a.zarr", {}, "", "same_section_x", ""),
        ("A_same_section", "ok_b.zarr", {}, "", "x", ""),
        ("P_NOLABEL", "nolabel.zarr", {"status": None}, "", "", ""),
        ("P_UNKNOWN", "unknown_status.zarr", {"status": "weird"}, "", "", ""),
        ("P_WRONG_KIND", "wrong_kind.zarr", {"kind": "other"}, "", "", ""),
        ("P_ABSENT_KIND", "absent_kind.zarr", {"kind": None}, "", "", ""),
        ("P_OTHER_DS", "other_ds.zarr", {"dataset_name": "SOMEONE_ELSE"}, "", "", ""),
        ("P_NO_DS", "no_ds.zarr", {"dataset_name": None}, "", "", ""),
        ("P_NO_GROUP", "no_group.zarr", {"fmt": "v2_no_zgroup"}, "", "", ""),
        ("P_ARRAY", "array.zarr", {"node_type": "array"}, "", "", ""),
        # landmark_min_used_pairs (default 6) must not exceed landmark_min_pairs.
        ("P_PAIRS_LOW", "pairs_low.zarr", {}, "landmark_min_pairs=5", "", ""),
        (
            "P_PAIRS_BOTH",
            "pairs_both.zarr",
            {"n_enabled": 4},
            "landmark_min_pairs=4;landmark_min_used_pairs=4",
            "",
            "",
        ),
        ("P_USED_HIGH", "used_high.zarr", {}, "landmark_min_used_pairs=9", "", ""),
        (
            "P_PAIRS_PRESET",
            "pairs_preset.zarr",
            {},
            "serial_section:landmark_min_pairs=5",
            "",
            "same_section;serial_section",
        ),
    ]
    rows: list[list[str]] = []
    for ident, store, kwargs, settings, label, presets in preflight_rows:
        if kwargs is not None:
            _fake_store(stores / store, **{"dataset_name": ident, **kwargs})
        rows.append(
            [
                ident,
                img("a.czi"),
                img("c.ome.tif"),
                str(stores / store),
                settings,
                label,
                presets,
            ]
        )
    p.add(
        "register_preflight",
        [*HEADER, "prepared_store", "registration_settings", "run_label", "presets"],
        rows,
        mode="register",
    )

    ok_rows = []
    for ident, store, kwargs, settings in [
        ("Q_OK", "q_ok.zarr", {}, ""),
        ("Q_V2", "q_v2.zarr", {"fmt": "v2"}, ""),
        (
            "Q_SIX",
            "q_six.zarr",
            {"n_enabled": 6, "n_disabled": 2},
            "landmark_min_pairs=6",
        ),
    ]:
        _fake_store(stores / store, dataset_name=ident, **kwargs)
        ok_rows.append(
            [ident, img("a.czi"), img("c.ome.tif"), str(stores / store), settings, ""]
        )
    p.add(
        "register_ok",
        [*HEADER, "prepared_store", "registration_settings", "presets"],
        ok_rows,
        mode="register",
    )

    # The global param lowers landmark_min_pairs only; a row can lower the other too.
    global_rows = []
    for ident, settings in [
        ("G_LOW", ""),
        ("G_ROW_FIXES", "landmark_min_used_pairs=5"),
    ]:
        store = _fake_store(stores / f"{ident}.zarr", dataset_name=ident)
        global_rows.append(
            [ident, img("a.czi"), img("c.ome.tif"), str(store), settings]
        )
    p.add(
        "register_pairs_global",
        [*HEADER, "prepared_store", "registration_settings"],
        global_rows,
        mode="register",
        params={"manual_registration_settings": "landmark_min_pairs=5"},
    )

    p.add(
        "configs",
        [*HEADER, "run_label", "registration_settings", "presets"],
        [
            [
                "X1",
                img("a.czi"),
                img("c.ome.tif"),
                "r1",
                "nucleus_qc=false",
                "serial_section;same_section",
            ]
        ],
        params={
            "manual_registration_settings": (
                "serial_section:nonrigid_smoothing_sigma_um=150"
            ),
        },
    )

    # Root attributes of stores.
    states = p.stores / "states"
    p.stores_cases = {
        "v3": str(_fake_store(states / "v3.zarr")),
        "v2": str(_fake_store(states / "v2.zarr", fmt="v2")),
        "v2_no_zgroup": str(
            _fake_store(states / "v2_no_zgroup.zarr", fmt="v2_no_zgroup")
        ),
        "v3_array": str(_fake_store(states / "v3_array.zarr", node_type="array")),
        "missing": str(states / "does_not_exist.zarr"),
        "no_metadata": str(_empty_dir(states / "empty.zarr")),
        "malformed": str(_malformed(states / "malformed.zarr")),
        "enabled_absent": str(
            _fake_store(
                states / "absent_enabled.zarr",
                n_enabled=5,
                n_disabled=2,
                drop_enabled_key=True,
            )
        ),
        "draft": str(_fake_store(states / "draft.zarr", status="draft")),
    }


def _empty_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _malformed(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "zarr.json").write_text("{not json")
    return path


SETTINGS_CASES: list[tuple[str, str | None]] = [
    ("a", "item 'a': expected [preset:]key=value"),
    ("foo:x=1", "unknown preset prefix 'foo' (allowed: same_section, serial_section)"),
    ("Bad=1", "key 'Bad' must match [a-z][a-z0-9_]*"),
    (
        "reference_channel=DAPI",
        "'reference_channel' cannot be overridden: set pixel sizes / "
        "registration channels in the samplesheet and re-run prepare",
    ),
    ("nope=1", "unknown key 'nope' (allowed: build_viewer_pyramid, chunk_size"),
    (
        "nonrigid_enabled=",
        "item 'nonrigid_enabled=': empty value for 'nonrigid_enabled'",
    ),
    ("landmark_min_pairs=2", "landmark_min_pairs must be an integer >= 3 (got '2')"),
    ("serial_section:nonrigid_smoothing_sigma_um=150;", None),
    ("nucleus_qc=true;;tile_size=512", "empty item"),
    (
        "nucleus_qc=true;nucleus_qc=false",
        "'nucleus_qc' is given twice for scope all presets",
    ),
    (
        "landmark_min_used_pairs=2",
        "landmark_min_used_pairs must be an integer >= 3 (got '2')",
    ),
]


# --- the launch ------------------------------------------------------------


def _once_per_session(
    tmp_path_factory: pytest.TempPathFactory,
    name: str,
    compute: Callable[[Path], object],
) -> Any:
    root = tmp_path_factory.getbasetemp()
    if os.environ.get("PYTEST_XDIST_WORKER"):
        root = root.parent
    directory = root / name
    result_file = root / f"{name}.json"
    with (root / f"{name}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not result_file.is_file():
            shutil.rmtree(directory, ignore_errors=True)
            directory.mkdir()
            result_file.write_text(json.dumps(compute(directory)))
    return json.loads(result_file.read_text())


def _launch(root: Path) -> dict[str, Any]:
    project = _Project(root)
    _build_cases(project)
    shutil.copy(
        WORKFLOWS / "lib" / "ManualRegistrationSheet.groovy", project.project / "lib"
    )
    (project.project / "lib" / "SheetHarness.groovy").write_text(HARNESS_CLASS)
    (project.project / "main.nf").write_text(HARNESS_MAIN)
    cases_file = root / "cases.json"
    cases_file.write_text(
        json.dumps({"parse": project.cases, "store": project.stores_cases})
    )
    out_file = root / "results.json"
    environment = {**os.environ, "NXF_DISABLE_CHECK_LATEST": "true"}
    environment["NXF_ANSI_LOG"] = "false"
    environment["NXF_OFFLINE"] = "true"
    environment["NXF_OPTS"] = f"{environment.get('NXF_OPTS', '')} {JVM_OPTIONS}".strip()
    assert NEXTFLOW is not None
    completed = subprocess.run(
        [
            NEXTFLOW,
            "-log",
            str(root / "nextflow.log"),
            "run",
            "main.nf",
            "--cases",
            str(cases_file),
            "--out",
            str(out_file),
        ],
        cwd=project.project,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    results = json.loads(out_file.read_text())
    results["_fs"] = str(project.fs)
    results["_project"] = str(project.project)
    return results


@pytest.fixture(scope="module")
def results(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    if NEXTFLOW is None:
        pytest.skip("nextflow is unavailable")
    return _once_per_session(tmp_path_factory, "manual_registration_sheet", _launch)


def _parse(results: dict[str, Any], name: str) -> dict[str, Any]:
    return results["parse"][name]  # type: ignore[no-any-return]


def _joined(results: dict[str, Any], name: str) -> str:
    return "\n".join(_parse(results, name)["errors"])


# --- parse: valid sheets ---------------------------------------------------


def test_minimal_row_takes_the_defaults(results: dict[str, Any]) -> None:
    parsed = _parse(results, "minimal")
    fs = results["_fs"]

    assert parsed["errors"] == [] and parsed["warnings"] == []
    (row,) = parsed["rows"]
    assert row["id"] == "X1" and row["row_number"] == 2
    assert row["presets"] == ["same_section"]
    assert row["reference"] == {
        "path": f"{fs}/img/a.czi",
        "scene": None,
        "channel_names": None,
        "registration_channel": "DAPI",
        "pixel_size_um": None,
    }
    assert row["moving"]["registration_channel"] == "DAPI"
    assert row["run_label"] is None and row["registration_settings"] == ""
    assert row["prepared_store"] == f"{fs}/out/X1/prepared/X1_prepared.zarr"


def test_relative_paths_resolve_against_the_sheet_folder(
    results: dict[str, Any],
) -> None:
    parsed = _parse(results, "relative_paths")
    img = Path(results["_fs"]) / "img"

    (row,) = parsed["rows"]
    assert parsed["errors"] == []
    assert row["reference"]["path"] == str(img / "a.czi")
    assert row["moving"]["path"] == str(img / "c.ome.tif")


def test_quoted_cells_bom_and_crlf_parse(results: dict[str, Any]) -> None:
    parsed = _parse(results, "quoted_bom_crlf")
    img = Path(results["_fs"]) / "img"

    assert parsed["errors"] == []
    (row,) = parsed["rows"]
    assert row["moving"]["channel_names"] == ["DAPI", "AF568"]
    assert row["moving"]["path"] == str(img / "with,comma" / "g.ome.tif")


def test_scene_on_a_czi_is_kept(results: dict[str, Any]) -> None:
    parsed = _parse(results, "scene_on_czi")

    assert parsed["errors"] == []
    assert parsed["rows"][0]["reference"]["scene"] == "ScanRegion0"


def test_presets_inherit_the_param(results: dict[str, Any]) -> None:
    parsed = _parse(results, "presets_inherit")

    assert parsed["errors"] == []
    assert parsed["rows"][0]["presets"] == ["same_section", "serial_section"]


# --- parse: header and row errors ------------------------------------------


def test_header_errors_stop_the_row_checks(results: dict[str, Any]) -> None:
    missing = _parse(results, "missing_column")
    unknown = _parse(results, "unknown_column")
    duplicate = _parse(results, "duplicate_header")

    assert missing["rows"] == [] and unknown["rows"] == [] and duplicate["rows"] == []
    assert missing["errors"] == [
        "samplesheet: missing required column(s): moving_image "
        "(required: id, reference_image, moving_image)"
    ]
    assert len(unknown["errors"]) == 1
    assert unknown["errors"][0].startswith(
        "samplesheet: unknown column(s): foo (allowed: id, reference_image, "
        "moving_image,"
    )
    assert duplicate["errors"] == ["samplesheet: column 'id' appears twice"]


def test_bad_and_duplicate_ids(results: dict[str, Any]) -> None:
    errors = _parse(results, "bad_and_duplicate_ids")["errors"]

    assert errors == [
        "row 3 (?): id 'bad id' must match ^[A-Za-z0-9][A-Za-z0-9_.-]*$",
        "row 4 (9-ok): id '9-ok' is also used in row 2",
    ]


def test_extensions_and_missing_images(results: dict[str, Any]) -> None:
    errors = _parse(results, "extensions_and_missing")["errors"]
    img = Path(results["_fs"]) / "img"

    assert errors == [
        "row 2 (X1): reference_image must end in .czi, .ome.tif, .ome.tiff, .tif or "
        f".tiff: {img / 'f.png'}",
        f"row 3 (X2): moving_image not found: {img / 'missing.czi'}",
    ]


def test_same_file_is_refused_also_through_a_symlink(results: dict[str, Any]) -> None:
    errors = _parse(results, "same_file")["errors"]

    assert errors == [
        "row 2 (X1): reference_image and moving_image are the same file",
        "row 3 (X2): reference_image and moving_image are the same file",
    ]


def test_scene_on_a_tiff_is_an_error(results: dict[str, Any]) -> None:
    assert _parse(results, "scene_on_tiff")["errors"] == [
        "row 2 (X1): reference_scene is only valid for a .czi reference_image"
    ]


def test_channel_names_must_be_non_empty_and_unique(results: dict[str, Any]) -> None:
    errors = _parse(results, "channel_names")["errors"]

    assert errors == [
        "row 2 (X1): reference_channel_names has an empty name",
        "row 3 (X2): moving_channel_names repeats 'dapi'",
    ]


def test_pixel_sizes_are_positive_numbers(results: dict[str, Any]) -> None:
    parsed = _parse(results, "pixel_size")

    assert parsed["errors"] == [
        "row 2 (X1): reference_pixel_size_um must be a number > 0 (got '0')",
        "row 3 (X2): reference_pixel_size_um must be a number > 0 (got '-1')",
        "row 4 (X3): reference_pixel_size_um must be a number > 0 (got 'abc')",
    ]
    assert parsed["rows"] == []  # rows with errors are dropped


def test_valid_pixel_size_is_parsed(results: dict[str, Any]) -> None:
    parsed = _parse(results, "pixel_size_ok")

    assert parsed["errors"] == []
    assert parsed["rows"][0]["moving"]["pixel_size_um"] == 0.34511
    assert parsed["rows"][0]["reference"]["pixel_size_um"] is None


def test_bad_presets(results: dict[str, Any]) -> None:
    assert _parse(results, "presets_bad")["errors"] == [
        "row 2 (X1): presets: unknown preset 'fast' "
        "(allowed: same_section, serial_section)",
        "row 3 (X2): presets: 'same_section' listed twice",
    ]


@pytest.mark.parametrize(("index", "case"), list(enumerate(SETTINGS_CASES)))
def test_settings_grammar(
    results: dict[str, Any], index: int, case: tuple[str, str | None]
) -> None:
    _item, expected = case
    parsed = _parse(results, f"settings_{index}")

    if expected is None:
        assert parsed["errors"] == []
    else:
        assert len(parsed["errors"]) == 1, parsed["errors"]
        assert parsed["errors"][0].startswith("row 2 (X1): registration_settings: ")
        assert expected in parsed["errors"][0]


def test_params_are_validated_even_when_rows_are_fine(results: dict[str, Any]) -> None:
    errors = _joined(results, "params_checks")

    assert (
        "--manual_registration_presets: unknown preset 'fast' "
        "(allowed: same_section, serial_section)" in errors
    )
    assert (
        "--manual_registration_force_prepare must be true or false (got 'maybe')"
        in errors
    )
    assert (
        "--manual_registration_settings: item 'serial_section:Bad=1': key 'Bad'"
        in errors
    )


def test_prepared_store_is_a_warning_in_prepare_mode(results: dict[str, Any]) -> None:
    parsed = _parse(results, "prepare_mode_prepared_store")
    fs = results["_fs"]

    assert parsed["errors"] == []
    assert parsed["warnings"] == [
        "row 2 (X1): prepared_store is read by the register workflow only; prepare "
        f"writes {fs}/out/X1/prepared/X1_prepared.zarr"
    ]
    assert (
        parsed["rows"][0]["prepared_store"] == f"{fs}/out/X1/prepared/X1_prepared.zarr"
    )


# --- the CZI runtime param -------------------------------------------------


def test_czi_row_without_the_runtime_param(results: dict[str, Any]) -> None:
    one = _parse(results, "czi_unset")["errors"]
    both = _parse(results, "czi_unset_both")["errors"]

    assert len(one) == 1
    assert one[0].startswith(
        "row 2 (X1): reference_image is a .czi file but "
        "--manual_registration_czi_python is not set"
    )
    assert "workflows/conf/site.config" in one[0] and "in_process" in one[0]
    assert len(both) == 1
    assert "reference_image and moving_image is a .czi file" in both[0]


def test_czi_param_value_checks(results: dict[str, Any]) -> None:
    fs = results["_fs"]

    assert _parse(results, "czi_relative")["errors"] == [
        "--manual_registration_czi_python must be an absolute path (got 'rel/python')"
    ]
    assert _parse(results, "czi_missing")["errors"] == [
        f"--manual_registration_czi_python not found: {fs}/no_such_python"
    ]
    assert _parse(results, "czi_quote")["errors"] == [
        "--manual_registration_czi_python must not contain a quote"
    ]


def test_czi_check_covers_rows_with_other_errors(results: dict[str, Any]) -> None:
    errors = _parse(results, "czi_unset_and_row_errors")["errors"]
    czi_text = (
        "is a .czi file but --manual_registration_czi_python is not set; set it in "
        "workflows/conf/site.config"
    )

    row_errors = [e for e in errors if czi_text not in e]
    assert row_errors == [
        "row 2 (X1): reference_pixel_size_um must be a number > 0 (got 'abc')",
        "row 3 (X2): reference_pixel_size_um must be a number > 0 (got 'abc')",
        "row 4 (?): id 'bad id' must match ^[A-Za-z0-9][A-Za-z0-9_.-]*$",
        "row 5 (X1): id 'X1' is also used in row 2",
    ]
    # One message per row that names a .czi: rows 2, 4 and 5; row 3 is TIFF only.
    czi_errors = [e for e in errors if czi_text in e]
    assert [e.split(":")[0] for e in czi_errors] == [
        "row 2 (X1)",
        "row 4 (?)",
        "row 5 (X1)",
    ]
    assert "reference_image is a .czi" in czi_errors[0]
    assert "reference_image and moving_image is a .czi" in czi_errors[1]


def test_row_numbers_are_physical_record_numbers(results: dict[str, Any]) -> None:
    img = Path(results["_fs"]) / "img"

    assert _parse(results, "blank_lines")["errors"] == [
        "row 3 (X1): reference_image must end in .czi, .ome.tif, .ome.tiff, .tif or "
        f".tiff: {img / 'f.png'}",
        "row 7 (X1): id 'X1' is also used in row 3",
    ]
    assert _parse(results, "blank_line_then_open_quote")["errors"] == [
        "samplesheet: unterminated quoted field starting in row 3"
    ]


@pytest.mark.parametrize(
    "case", ["outdir_default", "outdir_bare", "outdir_slash", "outdir_dots"]
)
def test_the_main_pipelines_results_folder_is_refused(
    results: dict[str, Any], case: str
) -> None:
    errors = _parse(results, case)["errors"]

    assert len(errors) == 1
    assert errors[0].startswith("--outdir '")
    assert errors[0].endswith(
        "pass --outdir with a folder separate from the main pipeline's results"
    )


def test_other_outdirs_are_accepted(results: dict[str, Any]) -> None:
    assert _parse(results, "outdir_other_relative")["errors"] == []
    assert _parse(results, "outdir_absolute_results")["errors"] == []
    assert _parse(results, "outdir_blank")["errors"] == ["--outdir is required"]


@pytest.mark.parametrize(
    "case", ["czi_existing", "czi_in_process", "czi_register_mode", "czi_tiff_only"]
)
def test_czi_param_accepted(results: dict[str, Any], case: str) -> None:
    assert _parse(results, case)["errors"] == []


def test_czi_python_helper(results: dict[str, Any]) -> None:
    assert (
        _parse(results, "czi_existing")["czi_python"] == f"{results['_fs']}/fake_python"
    )
    assert _parse(results, "czi_in_process")["czi_python"] == "in_process"


# --- the register preflight ------------------------------------------------


def test_register_preflight_messages(results: dict[str, Any]) -> None:
    parsed = _parse(results, "register_preflight")
    stores = Path(results["_fs"]) / "stores"
    assert parsed["errors"] == []
    messages = parsed["preflight"]

    def one(prefix: str) -> str:
        found = [m for m in messages if m.startswith(prefix)]
        assert len(found) == 1, (prefix, messages)
        return found[0]

    assert one("P_MISSING:") == (
        f"P_MISSING: prepared store not found: {stores / 'missing.zarr'}; run "
        "workflows/manual_registration_prepare.nf with this samplesheet and "
        "--outdir first"
    )
    assert one("P_DRAFT:") == (
        f"P_DRAFT: the landmarks in {stores / 'draft.zarr'} are a draft; open it "
        "in the "
        "viewer and press Finish labelling"
    )
    assert one("P_RESULT:") == (
        f"P_RESULT: {stores / 'result.zarr'} is a result store, not a prepared store; "
        "set prepared_store to the P_RESULT_prepared.zarr store"
    )
    assert one("P_NEWER:") == (
        f"P_NEWER: {stores / 'newer.zarr'} was written by a newer MerXen/viewer "
        "(schema_version 2 > 1); update"
    )
    assert one("P_SEVEN (preset same_section)") == (
        f"P_SEVEN (preset same_section): 7 enabled landmark pairs in "
        f"{stores / 'seven.zarr'}, "
        "at least 8 are needed"
    )
    assert one("P_NOLABEL:").startswith(
        f"P_NOLABEL: no landmarks saved in {stores / 'nolabel.zarr'}; open it in "
        "the viewer "
        "(Dataset loader > Manual Registration > Load image data)"
    )
    assert one("P_UNKNOWN:") == (
        f"P_UNKNOWN: unknown labelling status 'weird' in "
        f"{stores / 'unknown_status.zarr'}"
    )
    assert one("P_WRONG_KIND:") == (
        f"P_WRONG_KIND: {stores / 'wrong_kind.zarr'} is not a prepared "
        "manual-registration "
        "store (merxen_manual_registration_store.kind = other)"
    )
    assert one("P_ABSENT_KIND:").endswith("kind = missing)")
    # 6 enabled pairs: the row-wide override lowers the floor of every preset,
    # so P_SIX passes;
    # P_SIX_PRESET raises it for serial_section only (7), same_section keeps 8.
    assert not [
        m for m in messages if m.startswith("P_SIX:") or m.startswith("P_SIX (")
    ]
    assert [m for m in messages if m.startswith("P_SIX_PRESET")] == [
        f"P_SIX_PRESET (preset same_section): 6 enabled landmark pairs in "
        f"{stores / 'six_preset.zarr'}, at least 8 are needed",
        f"P_SIX_PRESET (preset serial_section): 6 enabled landmark pairs in "
        f"{stores / 'six_preset.zarr'}, at least 7 are needed",
    ]
    assert not [m for m in messages if m.startswith("P_V2")]
    assert one("P_OTHER_DS:") == (
        f"P_OTHER_DS: {stores / 'other_ds.zarr'} belongs to dataset SOMEONE_ELSE; "
        "prepared_store must be this id's store"
    )
    assert one("P_NO_DS:") == (
        f"P_NO_DS: {stores / 'no_ds.zarr'} records no dataset name; "
        "prepared_store must be this id's store"
    )
    assert one("P_NO_GROUP:") == (
        f"P_NO_GROUP: cannot read the root attributes of {stores / 'no_group.zarr'}: "
        "no zarr.json or .zgroup"
    )
    assert one("P_ARRAY:") == (
        f"P_ARRAY: cannot read the root attributes of {stores / 'array.zarr'}: "
        "zarr.json is not a Zarr group (node_type array)"
    )
    # landmark_min_used_pairs (default 6) must not exceed landmark_min_pairs, with the
    # precedence of the settings: 5 < 6 and 8 < 9 fail, 4 with 4 passes, and a preset
    # prefix lowers the floor of its preset only.
    too_few = (
        "landmark_min_pairs ({n}) is below landmark_min_used_pairs ({u}); "
        "landmark_min_used_pairs must not exceed landmark_min_pairs, so lower both "
        "together (landmark_min_used_pairs is 6 by default)"
    )
    assert [m for m in messages if m.startswith("P_PAIRS_LOW")] == [
        "P_PAIRS_LOW (preset same_section): " + too_few.format(n=5, u=6)
    ]
    assert [m for m in messages if m.startswith("P_USED_HIGH")] == [
        "P_USED_HIGH (preset same_section): " + too_few.format(n=8, u=9)
    ]
    assert [m for m in messages if m.startswith("P_PAIRS_PRESET")] == [
        "P_PAIRS_PRESET (preset serial_section): " + too_few.format(n=5, u=6)
    ]
    assert not [m for m in messages if m.startswith("P_PAIRS_BOTH")]
    assert one("run id ") == (
        "run id 'A_same_section_same_section_x' is used twice (rows 10 and 11); set "
        "run_label to tell the runs apart"
    )


def test_pair_counts_resolve_across_the_params_and_the_row(
    results: dict[str, Any],
) -> None:
    parsed = _parse(results, "register_pairs_global")

    assert parsed["errors"] == []
    (message,) = parsed["preflight"]
    assert message.startswith(
        "G_LOW (preset same_section): landmark_min_pairs (5) is below "
        "landmark_min_used_pairs (6)"
    )


def test_register_preflight_accepts_v2_and_a_lowered_floor(
    results: dict[str, Any],
) -> None:
    parsed = _parse(results, "register_ok")

    assert parsed["errors"] == []
    assert parsed["preflight"] == []
    assert [run["run_id"] for run in parsed["runs"]] == [
        "Q_OK_same_section",
        "Q_V2_same_section",
        "Q_SIX_same_section",
    ]


# --- store states ----------------------------------------------------------


def test_store_states(results: dict[str, Any]) -> None:
    store = results["store"]

    assert store["v3"] == {
        "exists": True,
        "kind": "prepared",
        "datasetName": "DS",
        "schemaVersion": 1,
        "status": "finished",
        "labellingSchemaVersion": 1,
        "nEnabledPairs": 8,
        "nPairs": 8,
        "error": None,
    }
    assert store["v2"]["exists"] and store["v2"]["error"] is None
    assert store["v2"]["nEnabledPairs"] == 8 and store["v2"]["kind"] == "prepared"
    assert store["v2"]["datasetName"] == "DS"
    assert store["draft"]["status"] == "draft"
    assert store["missing"] == {
        "exists": False,
        "kind": None,
        "datasetName": None,
        "schemaVersion": None,
        "status": None,
        "labellingSchemaVersion": None,
        "nEnabledPairs": 0,
        "nPairs": 0,
        "error": None,
    }
    assert store["no_metadata"]["exists"] is True
    assert store["no_metadata"]["error"] == "no zarr.json or .zgroup"
    # read_root_attrs refuses both: a v2 directory without .zgroup and a v3 array.
    assert store["v2_no_zgroup"]["error"] == "no zarr.json or .zgroup"
    assert store["v2_no_zgroup"]["kind"] is None
    assert store["v3_array"]["error"] == (
        "zarr.json is not a Zarr group (node_type array)"
    )
    assert store["v3_array"]["kind"] is None
    assert store["malformed"]["exists"] is True and store["malformed"]["error"]
    assert not store["malformed"]["error"].startswith("{")
    assert store["enabled_absent"]["nPairs"] == 7
    assert store["enabled_absent"]["nEnabledPairs"] == 5


# --- config JSON -----------------------------------------------------------


def test_emitted_configs_validate_in_python(results: dict[str, Any]) -> None:
    parsed = _parse(results, "configs")
    fs = results["_fs"]

    assert parsed["errors"] == []
    prepare = ManualRegistrationPrepareConfig.model_validate(
        json.loads(parsed["prepare"][0])
    )
    assert prepare.dataset_name == "X1" and prepare.force is False
    assert str(prepare.prepared_dir) == f"{fs}/out/X1/prepared"
    assert prepare.workers is None
    assert str(prepare.reference.path) == f"{fs}/img/a.czi"

    runs = {run["preset"]: run for run in parsed["runs"]}
    assert list(runs) == ["serial_section", "same_section"]
    run = runs["serial_section"]
    assert run["run_id"] == "X1_serial_section_r1"
    config = ManualRegistrationRunConfig.model_validate(json.loads(run["config_json"]))
    assert str(config.result_dir).endswith("/X1/registered/serial_section_r1")
    assert config.global_settings == "serial_section:nonrigid_smoothing_sigma_um=150"
    assert config.row_settings == "nucleus_qc=false"
    assert config.force is False and config.run_label == "r1"
    assert run["summary_path"].endswith(
        "/X1/registered/serial_section_r1/X1_serial_section_r1_manual_registration_summary.json"
    )
    assert run["result_store"].endswith(
        "/X1/registered/serial_section_r1/X1_serial_section_r1_registered.zarr"
    )

    viewer = ViewerCacheConfig.model_validate(json.loads(parsed["viewer"][0]))
    assert viewer.image_only is True
    assert viewer.image_keys == ["reference", "moving"]
    assert viewer.min_size == 1024 and viewer.downsample == 4

    compare = ManualRegistrationCompareConfig.model_validate(
        json.loads(parsed["compare"])
    )
    assert compare.dataset_name == "X1" and len(compare.summary_paths) == 2
    assert str(compare.output_dir) == f"{fs}/out/X1/registered"


def test_settings_keys_equal_the_model(results: dict[str, Any]) -> None:
    expected = sorted(
        set(ManualRegistrationSettings.model_fields) - FORBIDDEN_OVERRIDES
    )

    assert sorted(results["settings_keys"]) == expected


def test_allowed_columns(results: dict[str, Any]) -> None:
    assert results["allowed_columns"] == [
        "id",
        "reference_image",
        "moving_image",
        "reference_scene",
        "moving_scene",
        "reference_channel_names",
        "moving_channel_names",
        "reference_registration_channel",
        "moving_registration_channel",
        "reference_pixel_size_um",
        "moving_pixel_size_um",
        "presets",
        "registration_settings",
        "run_label",
        "prepared_store",
    ]


def test_names(results: dict[str, Any]) -> None:
    names = results["names"]

    assert names["run_name"] == "serial_section_trial"
    assert names["run_name_blank"] == "same_section"
    assert names["run_id"] == "LCM1_serial_section"
    assert (
        names["prepared_dir_relative"] == f"{results['_project']}/rel/out/LCM1/prepared"
    )
