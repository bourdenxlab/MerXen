# CZI test fixtures

Tiny synthetic Zeiss CZI files and an independent oracle for the CZI reader
(`src/merxen/io/czi.py`) and the `convert-czi` command. They hold no host or
sample information: the generator asserts that no file contains the output
path, a user name, a home directory or a host name.

| File | Content |
|---|---|
| `two_scenes_zstd.czi` | Two scenes (origins `(-1000, 500)` and `(2000, 3000)`), each a 2 x 2 tile mosaic with 8 px overlap, 72 x 56 px; 3 channels, 3 Z planes; Gray16, Zstd1; no channel names; 0.5 um pixels |
| `one_scene_hole.czi` | One scene at `(-131, 20)`, 96 x 80 px, three 48 x 40 tiles leaving a hole in the top right; 3 channels (`DAPI`, `AF568`, `AF647`), 1 Z plane; Gray16, uncompressed; 0.34511 um pixels |
| `rgb_bgr24.czi` | One 16 x 16 Bgr24 tile (rejected by the reader) |
| `time_t2.czi` | One 16 x 16 tile for each of two time points (rejected by the reader) |
| `no_scaling.czi` | One 16 x 16 tile without Scaling metadata |
| `oracle.npz` | `two_scenes_zstd__s0`, `two_scenes_zstd__s1`, `one_scene_hole__s0`: `(C, H, W)` uint16 composites; `two_scenes_zstd__bbox` and `one_scene_hole__bbox`: scene bounding boxes `(x, y, w, h)` |

The oracle is computed with numpy (tiles composited in write order, last
written wins; then the maximum over Z) and cross-checked by the generator
against a pylibCZIrw read of every scene, channel and plane.

## Regeneration

Only on purpose: the CZI bytes change on every run (pylibCZIrw writes random
GUIDs), the oracle does not.

```bash
uv run --no-project --python 3.12 --with pylibCZIrw==6.1.0 --with numpy==2.2.6 \
    python scripts/manual_registration/make_czi_fixtures.py --out tests/test_data/czi
```

`--check` regenerates into a temporary directory and compares the oracle with
the committed `oracle.npz`. pylibCZIrw (Zeiss, LGPL-3.0) is not a MerXen
dependency.
