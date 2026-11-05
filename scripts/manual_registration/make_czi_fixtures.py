#!/usr/bin/env python
"""Generate the tiny CZI test fixtures and their independent oracle.

Run once, and on purpose only, in an ephemeral environment (pylibCZIrw, Zeiss,
LGPL-3.0, is never a MerXen dependency)::

    uv run --no-project --python 3.12 --with pylibCZIrw==6.1.0 --with numpy==2.2.6 \\
        python scripts/manual_registration/make_czi_fixtures.py \\
        --out tests/test_data/czi

The CZI bytes differ on every run (pylibCZIrw writes random GUIDs); the oracle
(``oracle.npz``) does not: it is computed with numpy (tiles composited in
write order, last written wins, then the maximum over Z) and cross-checked
here against a pylibCZIrw read of every scene, channel and plane.
``--check`` regenerates into a temporary directory and compares the oracle
arrays with the committed ``oracle.npz``.

The files contain no host or sample information; the script asserts it.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import tempfile
from pathlib import Path

import numpy as np

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "tests" / "test_data" / "czi"
MAX_TOTAL_BYTES = 200 * 1000

ZSTD = "zstd1:ExplicitLevel=1;PreProcess=HiLoByteUnpack"

# two_scenes_zstd: scene origins in global pixels, 2 x 2 tile mosaic per scene.
TWO_SCENE_ORIGINS = {0: (-1000, 500), 1: (2000, 3000)}
TWO_TILE_ROWS, TWO_TILE_COLS = 32, 40
TWO_STEP_X, TWO_STEP_Y = 32, 24
TWO_C, TWO_Z = 3, 3

# one_scene_hole: tiles (x, y) in scene-local pixels; the top-right block is a hole.
HOLE_ORIGIN = (-131, 20)
HOLE_TILE_ROWS, HOLE_TILE_COLS = 40, 48
HOLE_TILES = [(0, 0), (0, 40), (48, 40)]
HOLE_C = 3


def _grid(rows: int, cols: int, y0: int, x0: int) -> tuple[np.ndarray, np.ndarray]:
    """Scene-local ``(y, x)`` coordinates of a ``rows x cols`` tile at ``(y0, x0)``."""
    y, x = np.mgrid[0:rows, 0:cols].astype(np.int64)
    return y + y0, x + x0


def two_scene_tile(s: int, c: int, z: int, ty: int, tx: int) -> np.ndarray:
    """Pixels of one tile of ``two_scenes_zstd`` (uint16, rows x cols)."""
    y, x = _grid(TWO_TILE_ROWS, TWO_TILE_COLS, TWO_STEP_Y * ty, TWO_STEP_X * tx)
    tile_id = 2 * ty + tx
    value = 1000 * c + 10 * s + tile_id + 50 * ((3 * x + 5 * y + 7 * z) % 4)
    return value.astype(np.uint16)


def hole_tile(c: int, i: int, x0: int, y0: int) -> np.ndarray:
    """Pixels of tile ``i`` of ``one_scene_hole`` (uint16, rows x cols)."""
    y, x = _grid(HOLE_TILE_ROWS, HOLE_TILE_COLS, y0, x0)
    value = 2000 * c + 100 * i + (x + 2 * y) % 97
    return value.astype(np.uint16)


def oracle_two_scenes() -> dict[str, np.ndarray]:
    """Numpy composite (last written wins), max over Z, per scene: (C, 56, 72)."""
    out = {}
    for s in TWO_SCENE_ORIGINS:
        volume = np.zeros((TWO_C, TWO_Z, 56, 72), np.uint16)
        for c in range(TWO_C):
            for z in range(TWO_Z):
                for ty in range(2):
                    for tx in range(2):
                        y0, x0 = TWO_STEP_Y * ty, TWO_STEP_X * tx
                        volume[
                            c, z, y0 : y0 + TWO_TILE_ROWS, x0 : x0 + TWO_TILE_COLS
                        ] = two_scene_tile(s, c, z, ty, tx)
        out[f"two_scenes_zstd__s{s}"] = volume.max(axis=1)
    return out


def oracle_hole() -> dict[str, np.ndarray]:
    """Numpy composite of ``one_scene_hole``: (C, 80, 96), the hole stays 0."""
    volume = np.zeros((HOLE_C, 80, 96), np.uint16)
    for c in range(HOLE_C):
        for i, (x0, y0) in enumerate(HOLE_TILES):
            volume[c, y0 : y0 + HOLE_TILE_ROWS, x0 : x0 + HOLE_TILE_COLS] = hole_tile(
                c, i, x0, y0
            )
    return {"one_scene_hole__s0": volume}


def oracle_arrays() -> dict[str, np.ndarray]:
    """Every array of ``oracle.npz``."""
    arrays: dict[str, np.ndarray] = {}
    arrays.update(oracle_two_scenes())
    arrays.update(oracle_hole())
    arrays["two_scenes_zstd__bbox"] = np.array(
        [[-1000, 500, 72, 56], [2000, 3000, 72, 56]], dtype=np.int64
    )
    arrays["one_scene_hole__bbox"] = np.array([[-131, 20, 96, 80]], dtype=np.int64)
    return arrays


def _plane(tile: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(tile[:, :, None])


def write_two_scenes(out: Path) -> None:
    from pylibCZIrw import czi as pyczi

    with pyczi.create_czi(
        str(out / "two_scenes_zstd.czi"), exist_ok=True, compression_options=ZSTD
    ) as doc:
        for s, (ox, oy) in TWO_SCENE_ORIGINS.items():
            for c in range(TWO_C):
                for z in range(TWO_Z):
                    for ty in range(2):
                        for tx in range(2):
                            doc.write(
                                _plane(two_scene_tile(s, c, z, ty, tx)),
                                location=(
                                    ox + TWO_STEP_X * tx,
                                    oy + TWO_STEP_Y * ty,
                                ),
                                plane={"C": c, "Z": z, "T": 0},
                                scene=s,
                            )
        doc.write_metadata(
            document_name="fixture", scale_x=0.5e-6, scale_y=0.5e-6, scale_z=1e-6
        )


def write_hole(out: Path) -> None:
    from pylibCZIrw import czi as pyczi

    with pyczi.create_czi(str(out / "one_scene_hole.czi"), exist_ok=True) as doc:
        ox, oy = HOLE_ORIGIN
        for c in range(HOLE_C):
            for i, (x0, y0) in enumerate(HOLE_TILES):
                doc.write(
                    _plane(hole_tile(c, i, x0, y0)),
                    location=(ox + x0, oy + y0),
                    plane={"C": c, "Z": 0, "T": 0},
                    scene=0,
                )
        doc.write_metadata(
            document_name="fixture",
            channel_names={0: "DAPI", 1: "AF568", 2: "AF647"},
            scale_x=0.34511e-6,
            scale_y=0.34511e-6,
        )


def write_rgb(out: Path) -> None:
    from pylibCZIrw import czi as pyczi

    with pyczi.create_czi(str(out / "rgb_bgr24.czi"), exist_ok=True) as doc:
        doc.write(np.full((16, 16, 3), 7, np.uint8), plane={"C": 0}, location=(0, 0))
        doc.write_metadata(
            document_name="fixture",
            channel_names={0: "TL"},
            scale_x=0.5e-6,
            scale_y=0.5e-6,
        )


def write_time(out: Path) -> None:
    from pylibCZIrw import czi as pyczi

    with pyczi.create_czi(str(out / "time_t2.czi"), exist_ok=True) as doc:
        for t in range(2):
            doc.write(
                np.full((16, 16, 1), t, np.uint16),
                plane={"C": 0, "T": t},
                location=(0, 0),
            )
        doc.write_metadata(document_name="fixture", scale_x=0.5e-6, scale_y=0.5e-6)


def write_no_scaling(out: Path) -> None:
    from pylibCZIrw import czi as pyczi

    with pyczi.create_czi(str(out / "no_scaling.czi"), exist_ok=True) as doc:
        doc.write(np.full((16, 16, 1), 3, np.uint16), plane={"C": 0}, location=(0, 0))
        doc.write_metadata(document_name="fixture", channel_names={0: "DAPI"})


def cross_check(out: Path, oracle: dict[str, np.ndarray]) -> None:
    """Assert pylibCZIrw reads every scene, channel and plane as the numpy composite."""
    from pylibCZIrw import czi as pyczi

    for name, n_scenes in (("two_scenes_zstd", 2), ("one_scene_hole", 1)):
        with pyczi.open_czi(str(out / f"{name}.czi")) as doc:
            rects = doc.scenes_bounding_rectangle
            bbox = oracle[f"{name}__bbox"]
            assert len(rects) == n_scenes, (name, rects)
            z_range = doc.total_bounding_box.get("Z", (0, 1))
            for s in range(n_scenes):
                r = rects[s]
                assert [r.x, r.y, r.w, r.h] == bbox[s].tolist(), (name, s, r)
                expected = oracle[f"{name}__s{s}"]
                for c in range(expected.shape[0]):
                    planes = [
                        doc.read(
                            roi=(r.x, r.y, r.w, r.h),
                            plane={"C": c, "Z": z, "T": 0},
                            scene=s,
                        )[..., 0]
                        for z in range(z_range[0], z_range[1])
                    ]
                    got = np.maximum.reduce(planes)
                    assert np.array_equal(got, expected[c]), (name, s, c)


def assert_private(out: Path) -> None:
    """Assert no fixture holds a host path, user name, home directory or host name."""
    needles = {
        str(out.resolve()),
        os.environ.get("USER", ""),
        os.environ.get("HOME", ""),
        socket.gethostname(),
    }
    needles = {n for n in needles if len(n) >= 4}
    for path in sorted(out.glob("*.czi")):
        data = path.read_bytes()
        for needle in needles:
            assert needle.encode() not in data, f"{path.name} contains host information"


def generate(out: Path) -> dict[str, np.ndarray]:
    """Write all fixtures into ``out`` and return the oracle arrays."""
    out.mkdir(parents=True, exist_ok=True)
    oracle = oracle_arrays()
    write_two_scenes(out)
    write_hole(out)
    write_rgb(out)
    write_time(out)
    write_no_scaling(out)
    cross_check(out, oracle)
    assert_private(out)
    return oracle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate into a temp directory and compare with the committed oracle",
    )
    args = parser.parse_args(argv)
    if args.check:
        with tempfile.TemporaryDirectory(prefix="czi-fixtures-") as tmp:
            oracle = generate(Path(tmp))
        committed = np.load(DEFAULT_OUT / "oracle.npz")
        bad = [
            name
            for name in sorted(set(oracle) | set(committed.files))
            if name not in oracle
            or name not in committed.files
            or not np.array_equal(oracle[name], committed[name])
        ]
        if bad:
            print(f"oracle differs from the committed one: {bad}", file=sys.stderr)
            return 1
        print("oracle matches the committed oracle.npz")
        return 0
    oracle = generate(args.out)
    np.savez_compressed(args.out / "oracle.npz", **oracle)
    total = sum(p.stat().st_size for p in args.out.iterdir() if p.is_file())
    assert total < MAX_TOTAL_BYTES, f"fixtures take {total} bytes"
    print(
        f"wrote {len(list(args.out.glob('*.czi')))} CZI files and oracle.npz "
        f"({total} bytes)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
