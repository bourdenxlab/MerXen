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
python -m merxen.annotation.prediction_metadata \
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
