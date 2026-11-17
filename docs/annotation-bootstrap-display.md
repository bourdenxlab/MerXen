# Exploring saved bootstrap predictions

The whole-panel hippocampal profile keeps raw atlas predictions separately from
accepted annotations. An unresolved accepted subclass or supercluster can still
have a raw prediction, including a unanimous bootstrap result. Acceptance also
depends on class/depth calibration, reproduced diagnostic rules and accepted
parents. A display cut-off must not silently change that acceptance policy.

Observation fields use `ct_seaad_subclass_*` and `ct_whb_supercluster_*`:

- `raw_name`, `raw_assignment`: the mapper's original chosen type and stable ID.
- `bp`: bootstrap vote fraction at this step, conditional on preceding choices.
- `aggregate_probability`: the mapper's product of support along the chosen
  hierarchy; despite its upstream name, this is not a calibrated biological
  probability.
- `avg_correlation`: the mapper's average correlation for the chosen type.
- `runner_up_1_*` through `runner_up_5_*`: saved alternative IDs, names,
  `probability` (bootstrap vote fraction), and correlation. A display of five
  candidates uses the winner plus up to four alternatives; fewer are available
  when fewer types received bootstrap support. These are conditional candidates,
  not a global posterior over every atlas type. Do not renormalize the displayed
  subset or invent zero-support alternatives.
- `name`, `assignment`, `accepted`, `resolution_status`: the separate existing
  acceptance decision and its explanation.

The viewer offers **SEA-AD · subclass (all predictions)** and **Allen · supercluster
(all predictions)** alongside accepted annotations. Their minimum-bootstrap
control starts at zero. Increasing it filters the raw vote fraction (`bp`), with
below-cut-off cells grey in UMAP and unfilled in the spatial overlay. Both views
share the cut-off. The cell inspector retains candidates, hierarchy support and
acceptance status regardless of the display cut-off. Zero-expression objects
remain without expression-based predictions.

New RESOLVE outputs include these fields in `celltype_labels.parquet` for
downstream table publication. Earlier runs already retain the alternatives in
`seaad_resolved.parquet` and `whb_resolved.parquet`; no mapping rerun is needed.
To enrich an existing store's explicitly chosen tables, close readers and run:

```bash
python -m merxen.io.annotation_predictions \
  --store "$MERXEN_VIEWER_STORE" \
  --table "$MERXEN_VIEWER_TABLE" \
  --seaad-resolved "$MERXEN_SEAAD_RESOLVED" \
  --whb-resolved "$MERXEN_WHB_RESOLVED"
```

Repeat `--table` for other tables from the same segmentation. Run separately for
each segmentation with its own resolved files. This operation checks exact cell
IDs, refuses conflicting existing prediction fields, takes the store write lock,
and changes only observation groups and consolidated metadata. It preserves
accepted labels, expression matrices, embeddings, shapes, images and transcript
assignments. It is repeatable with identical predictions and rolls back earlier
observation replacements if a later replacement fails. Use a versioned export
when an existing download package must remain immutable.

## Map-first runs

A `map_first` run (frontal cortex, or the provisional hippocampal formation)
keeps the same raw predictions in MAP's tidy files under `annotation_map_out`:
`<sample>_mmc_seaad_mr_panel.parquet` and `<sample>_mmc_<WHB primary>.parquet`
(`whb_frontal_supc_clus` or `whb_hippocampal_supc_clus`), five runners-up at
every level. The same command reads them, with the RESOLVE label table
(required) for the acceptance fields:

```bash
python -m merxen.io.annotation_predictions \
  --store "$MERXEN_VIEWER_STORE" \
  --table "$MERXEN_VIEWER_TABLE" \
  --seaad-mmc "$MERXEN_SEAAD_MMC" \
  --whb-mmc "$MERXEN_WHB_MMC" \
  --labels "$MERXEN_CELLTYPE_LABELS"
```

Here `bp` is MapMyCells' bootstrap fraction at that level, conditional on the
parent choice; `aggregate_probability` is the product along the hierarchy, the
value map-first RESOLVE thresholds for the SEA-AD subclass (`ct_seaad_subclass_raw`).

Both `ct_seaad_subclass_*` and `ct_whb_supercluster_*` get the full field set
of the whole-panel route. The acceptance fields come from the label table's
`ct_seaad_subclass_*` and `ct_supercluster_*` columns, with two differences
from the whole-panel route:

- `name` is the label table's call at that level, and map-first keeps it for
  cells it did not accept (the whole-panel `name` is empty there). Read it
  with `accepted`, which is true exactly when the status is `confident`.
  `assignment` is that name's node ID (from the tidy rows' calls and
  runners-up). For SEA-AD this `name` is the column the map-first hierarchy
  step already copied into the store, so it is not changed; a label table
  whose names differ from the store's is refused as a conflict.
- `resolution_status` is the label-table status, from the map-first
  vocabulary: `confident`, `low_confidence`, `below_floor`, `not_resolvable`,
  `method_disagree`, `single_method`, `implausible`, `parent_unresolved`,
  `not_attempted_gate`, `not_applicable`, `low_counts`. Whole-panel stores
  hold their own resolution statuses (`accepted_existing_atlas_policy`, ...).

The viewer code is not in this repository; a viewer that colours accepted
annotations by `name` alone must also read `accepted` for a map-first store.

Unlike the whole-panel files, the tidy files may not cover every table cell
(an atlas refused for the panel, or cells outside the mapped table): those
observations keep missing prediction fields instead of being refused. Run it
per sample and segmentation, and again after a later `FINALIZE` rewrites the
table.
