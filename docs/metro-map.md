# Metro map

The pipeline overview metro map is generated from [assets/metro_map.mmd](../assets/metro_map.mmd)
using [nf-metro](https://github.com/pinin4fjords/nf-metro), following the same
approach as [nf-core/rnaseq](https://github.com/nf-core/rnaseq/blob/master/docs/dev/metro_map.md).
Two SVGs are committed:

| File | Used by |
|------|---------|
| [docs/images/merxen_metro_map.svg](images/merxen_metro_map.svg) | [Documentation index](index.md), [Pipeline architecture](pipeline.md) |
| [docs/images/merxen_metro_map_animated.svg](images/merxen_metro_map_animated.svg) | [README](../README.md) |

Both are rendered with the light theme and keep nf-metro's
`prefers-color-scheme` block, so a single file adapts to the reader's GitHub
theme. The animation is baked-in CSS keyframes — no JavaScript and no SMIL — so
it plays inside a plain `<img>` on GitHub.

## What the lines mean

| Line | Colour | Meaning |
|------|--------|---------|
| MERSCOPE (Vizgen) | `#8350FF` | The MERSCOPE section's own traversal of the per-platform stages |
| Xenium (10x) | `#17BCAC` | The Xenium section's traversal of the same stages |
| Paired-section analysis | `#3980BE` | The single merged track, from `ALIGN` onwards, that consumes both platforms |
| Toggleable stages | `#EE7733`, dashed | Stages that can be switched on or off per run or per samplesheet row |

The two platform lines run through the same stations because both platforms
traverse `BUILD_SPATIALDATA → SEGMENT → ENRICH → MASK_IMAGE_QUANTIFICATION → QC`
independently. They converge at `ALIGN`, which is where a paired run stops
being two datasets and starts being one comparison.

"Toggleable" means the stage is governed by a parameter, **not** that it is off
by default. `MECR` and the ProSeg hybrid branch are dashed but run in a default
invocation (`mecr_enabled = true`, `proseg_hybrid_enabled = true`); cortical
depth, distance-from-object and MapMyCells are dashed and default to off. See
[Configuration](configuration.md) for the switches.

### Colour choice

The lines use the MerXen brand palette defined in
[src/merxen/palette.py](../src/merxen/palette.py), which is derived from the
title logo: MERSCOPE is the purple of "Mer", Xenium the teal of "Xen", and the
paired line the blue midpoint of the logo's underline gradient, where the two
halves meet. The toggleable line's orange sits deliberately outside that range
so it never reads as a platform.

`tests/test_workflows/test_metro_map.py` checks the `.mmd` line colours against
`merxen.palette` and fails if the two drift apart, so a rebrand is one edit
rather than a hunt. Change the palette module, re-run
`scripts/render_metro_map.py`, and commit the re-rendered SVGs.

> [!NOTE]
> This palette is chosen for brand consistency, not colour-vision
> accessibility. The MERSCOPE purple and Xenium teal are not reliably separable
> under simulated tritanopia (CIE76 ΔE ≈ 15) or deuteranopia (ΔE ≈ 20), and the
> teal is close to the section background under protanopia. An earlier palette
> guaranteed ≥30 ΔE under all three, and the test that enforced it was removed
> along with it. The toggleable line keeps its dash pattern as a redundant,
> colour-independent cue; the two platform lines have no such fallback, so the
> map relies on its station labels to be read without colour.

## What the map simplifies

A metro map is a reader's overview, not a DAG dump. The precise graph, including
everything below, stays in [Pipeline architecture](pipeline.md).

- **Segmentation branches are not drawn.** Downstream analysis runs separately
  for resegmented, original and (optionally) hybrid cells. Drawing that fan-out
  would triple every station after `ENRICH`.
- **Per-row multiplicity is not drawn.** Platform-local stations run once per
  platform per samplesheet row; the map shows one traversal.
- **Single-platform mode is not drawn.** With `--analysis_mode merscope` or
  `xenium`, one platform line is absent and the paired-only `ALIGN`, `ALIGN_QC`
  and `COMPARE` stations do not run, while `VISUALIZE`, `SPATIAL_GENE_ANALYSIS`
  and `CLUSTERING_SQUIDPY` run with a one-sample config.
- **Three processes have no station**, because they carry no scientific
  meaning for a reader: `ENSURE_PROSEG` (installs the ProSeg binary),
  `VIEWER_CACHE` (derived viewer cache) and `VALIDATE_ANALYSIS_LAYER` (a pre-QC
  guard producing no artifacts of its own).
- **One station stands for three processes.** `Clustering` covers
  `CLUSTERING_SQUIDPY_PREPARE`, `CLUSTERING_SQUIDPY_COMPUTE` and
  `CLUSTERING_SQUIDPY_FINALIZE`.
- **`ProSeg hybrid` is a station but not a process.** It is a branch of
  `PROSEG_SEGMENT` selected by `proseg_hybrid_enabled`.

Station labels are shortened to keep the map legible — `Spatial genes` is
`SPATIAL_GENE_ANALYSIS`, `Mask image quant.` is `MASK_IMAGE_QUANTIFICATION`.
Station *ids* in the `.mmd` always match the lowercased process name, which is
what the drift test keys on.

## Regenerating

Update the `.mmd` source whenever you add or rename a pipeline stage, then:

```bash
python scripts/render_metro_map.py
```

The script validates the source, renders both SVGs with
`--validate --strict` (a geometry defect fails the render rather than producing
a subtly broken picture), and appends the trailing newline that pre-commit's
`end-of-file-fixer` requires. It needs `nf-metro`, which ships in the dev extra:

```bash
uv pip install -e '.[dev]'
```

## Staying in sync

`tests/test_workflows/test_metro_map.py` runs in the normal `pytest` suite and
fails if:

- a Nextflow process under `workflows/modules/` has no station and is not on the
  omission list,
- a station names a process that no longer exists,
- the omission or alias lists name processes that have been deleted,
- a line colour drifts from `merxen.palette`, or two lines share a colour,
- a committed SVG is missing or was rendered from a different palette,
- the rendered figure grows past 2000px wide or a 3:1 aspect ratio,
- a station label wraps mid-word in the rendered SVG.

The test parses the `.mmd` with plain regexes and does not import `nf_metro`, so
it passes without the dev extra installed.

## Layout

The map is a serpentine rather than one long strip, because it is embedded at
the width of a README or docs column: a very wide, very short figure scales
down until nothing is readable.

- **Top row, left to right:** Ingest and build → Segmentation → Enrichment and
  QC.
- **At QC the trunk drops** to the second row and doubles back **right to
  left**: Alignment → Comparative analysis → Annotation and DE.
- **The MECR spur leaves QC to the right** and runs **top to bottom**, tucked
  into column 2 beneath Alignment.

Column 2 holds an `RL` section and a `TB` one, so it is right-aligned and
Enrichment and QC, Alignment and MECR rates share a right edge to
within a pixel. Ingest and build and Annotation and DE share a left edge in
column 0, but only because the render script forces it — see
[Styling](#styling).

`test_rendered_map_stays_compact` fails if the rendered SVG exceeds 2000px wide
or a 3:1 aspect ratio, so the figure cannot quietly stretch back out.

### Styling

nf-metro exposes no CLI flag for stroke weight or legend sizing, so
`scripts/render_metro_map.py` patches the light theme in place and then
delegates to nf-metro's own CLI. It overrides:

| Value | Default | Ours | Why |
|---|---|---|---|
| `line_width` | 3.0 | 5.5 | The map is scaled down when embedded; 3.0 strokes disappear |
| `legend_font_size` | 16.0 | 19.0 | Before `--font-scale`, so the legend reads at 26.6px |
| `LEGEND_SWATCH_WIDTH` | 24.0 | 56.0 | Longer swatches make the dash pattern legible |
| `LEGEND_LINE_HEIGHT` | 24.0 | 40.0 | Not scaled by `--font-scale`, so 24.0 is tighter than the 26.6px type is tall |
| `STROKE_DASHARRAY["dashed"]` | `8,4` | `12,10` | See below |
| `_pack_cells` | right-aligns | left edges aligned | See below |

The dash override is not cosmetic. nf-metro's `8,4` assumes the default 3.0
stroke, and round line caps extend half the stroke width past each dash end. At
5.5 the caps close the 4px gap completely and the toggleable line renders
**solid**, silently destroying the only colour-independent cue on the map. Any
future change to `line_width` must re-check the dash.

The `_pack_cells` override fixes the column-0 alignment. nf-metro right-aligns
every section in a column as soon as one of them runs `RL` or `TB`, which the
return row forces here. Ingest and build and Annotation and DE are different
widths, so a shared right edge left the wider Annotation box — and the
bottom-left legend under it — jutting out to the left of everything else.
`align_column_left_edges` wraps `_pack_cells`, the last thing
`_compute_section_offsets` does, and rewrites the offsets so both boxes share
their *drawn* left edge. It aligns `offset_x + bbox_x`, not `offset_x`: the
terminus icons on Ingest and build push that section's bbox further left, and
aligning the raw offsets leaves the boxes 8px apart. Annotation moves right to
meet Ingest rather than the other way round, because the return row already has
slack next to Comparative analysis while the top row does not.

`station_radius` is deliberately left at its 6.0 default: enlarging it to match
the heavier strokes widens each station enough to hyphenate `Spatial genes` and
`Clustering`, for no real gain.

These are nf-metro internals, which upstream documents as not semver-stable,
hence the `nf-metro>=1.1,<1.2` pin. The script raises rather than silently
rendering unstyled output if a name moves.

### Editing notes

Constraints worth knowing before you rearrange sections, all learned the hard
way against nf-metro 1.1.0:

- Routing a dashed line through a section's `bottom` port raises
  `CurveInvariantError` and aborts the render. Only the solid trunk uses the
  vertical hop between rows; the dashed MECR spur leaves sideways and descends
  outside any section. Use `left`/`right` ports and `%%metro grid:` placement
  if you need a dashed line to change row.
- A section header is drawn above its box **only when no route crosses that
  band**, and the band scales with `--font-scale`. At 1.4 the Alignment section
  tolerates about 14 characters and MECR rates about 11 before the band overruns
  into the descending trunk and nf-metro pushes the header to the side or below
  the box. This is why those two sections are not called "Section alignment" and
  "Co-expression" — both were over the limit. Widening the box is not an
  available fix: box width is content-driven, and sections are right-aligned in
  their column rather than stretched to fill it.
- Sections are **right-aligned** within a grid column whenever any section in
  that column runs `RL` or `TB`. Column 0 is exempted by the `_pack_cells`
  override described in [Styling](#styling), so Ingest and build and Annotation
  and DE keep a shared left edge at any width. Every other column is still
  right-aligned, and widening a section there moves its left edge, not its
  right.
- The gap between Comparative analysis and Annotation and DE is slack in column
  1, which is sized by Segmentation, the widest section. It can be moved
  elsewhere by re-columning but not removed without narrowing Segmentation.
- MECR rates is entered from the **right**, not the top, even though it sits
  directly below Alignment. A top entry routes the spur straight
  through the Alignment box without touching a station there, which
  `--validate --strict` rejects (`_guard_no_route_through_section`). Entering
  from the right sends it around the outside instead. This is exactly the class
  of defect that renders without complaint when validation is off, so do not
  drop those flags while experimenting.
- `--x-spacing` and `--font-scale` are not independent, and the intuitive move
  is the wrong one: **widening the spacing makes the on-screen text smaller**,
  because the figure grows faster than the type does. The narrowest wrap-free
  configuration wins. At the current labels that is `--x-spacing 90
  --font-scale 1.4`; raising the font further hyphenates `Cellpose-SAM cells`.
  Sweep both together and check `test_station_labels_are_not_wrapped` rather
  than nudging one value.
