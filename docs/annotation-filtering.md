# Filtering cells by confidence

Since label table version 2 ([always-label
plan](plans/annotation-always-label-plan.md)) map-first annotation labels
every cell its own evidence supports, for every sample, segmentation and
panel. The dataset gate, the panel's trust state and real-data QC no longer
remove labels. What they found is written into each cell's row instead, and
the answer an enforced gate would have given is kept in the `*_strict`
columns. Choosing which cells to trust is therefore a filter the analysis
applies, not something the pipeline decides.

This page lists the columns a filter reads and gives recipes, in plain
pandas and with `merxen.annotation.filtering.select_cells`. The method behind
each column is in the [annotation overview](annotation-overview.md); the full
column list is the [label table
schema](outputs.md#label-table-schema-sid_celltype_labelsparquet).

## Where the columns are

- **Label table**: `<pair>/<seg>/annotation_resolve/annotation_resolve_out/<platform>/<sid>_celltype_labels.parquet`
  under the run's `outdir`, one row per segmented object.
  `merxen.annotation.pipeline.read_label_table(path)` returns the table and
  its provenance.
- **Clustered table**: the `obs` of `<sid>_clustered.h5ad` and of the
  clustered SpatialData table holds the same `ct_*`, `flag_*`, `qc_*` and
  `*_strict` columns, `exclude_hard` and `exclude_hard_strict`, for table
  cells only. It has no `species` column; `select_cells` reads the species
  from its leaf status column (`ct_supercluster_status` or
  `ct_subclass_status`).

Every recipe below works on either.

## What a filter can read

| Question | Columns | Constant within a dataset? |
|---|---|---|
| Did this cell pass every check at level L? | `ct_<L>_status` (`confident`, or the reason it is not) | no |
| What is its deepest confident label? | `ct_final_level`, `ct_final_name` | no |
| Is the cell unusable whatever the level? | `exclude_hard` (off-table, or implausible at every level it was called) | no |
| Does a per-cell warning apply? | `flag_*` (e.g. `flag_microglial_spillover`, `flag_contaminated`, `flag_low_counts`); `discovery_caution` | no |
| How deep does the dataset's QC support labels? | `qc_dataset_max_level` (a `ct_final_level` value, `none` when the gate failed or the primary panel is refused or missing) | yes |
| What did the dataset gate say? | `qc_dataset_level` (`full`, `broad_only`, `failed`; advisory) | yes |
| Which dataset signals warned or tripped a fail rule? | `qc_dataset_warnings` (`;`-joined tokens ending in `_fail` or `_warn`) | yes |
| How far is the panel trusted? | `qc_panel_trust` (`validated_real`, `validated_simulation`, `provisional`, `broad_only`, `refused`, `unknown`) | yes |
| What were the signal values? | `qc_<signal>`: human `qc_frac_ge30`, `qc_broad_coverage_table`, `qc_broad_coverage_segmented`; mouse `qc_g1_density_ratio`, `qc_g1_shift_um`, `qc_g2_marker_consistency`, `qc_g3_implausible_share`, `qc_g4_astro_epen_points`, `qc_g4_immune_points`, `qc_g5_spillover_rate` | yes |
| What would version 1 (an enforced gate) have kept? | `ct_<L>_status_strict`, `ct_final_level_strict`, `ct_final_name_strict`, `exclude_hard_strict` | no |

The chain levels are human `lineage`, `broad`, `nt`, `supercluster` and
mouse `broad`, `class`, `nt`, `subclass`; outside the chain are the human
`seaad_subclass` (and opt-in fine levels). A level's name column
(`ct_<L>_name`) also holds the candidate name of a cell that failed a check,
so always filter on `ct_<L>_status`, not on a non-null name.

## Recipes in plain pandas

`labels` is a label table or a clustered `obs`. The `qc_*` and status
columns are categoricals; compare them as strings.

**Confident labels at one level.** The cell's own evidence only, as the
pipeline's hierarchy, MENDER states and default report use it:

```python
keep = labels["ct_subclass_status"].astype(str) == "confident"
keep &= ~labels["exclude_hard"]
```

**Leave out datasets with any fail-rule signal.**

```python
warnings = labels["qc_dataset_warnings"].astype(str)
keep &= ~warnings.str.contains("_fail", regex=False)
```

**Only levels the dataset's QC supports.** Compare ranks along the
species' chain; `merxen.annotation.vocab.FINAL_LEVELS` holds the order:

```python
from merxen.annotation.vocab import FINAL_LEVELS

chain = FINAL_LEVELS["mouse"]  # ("none", "broad", "class", "nt", "subclass")
rank = {level: index for index, level in enumerate(chain)}
supported = labels["qc_dataset_max_level"].astype(str).map(rank)
keep &= supported >= rank["subclass"]
```

**Your own cut-off on a dataset signal**, for example a stricter
registration ratio. A signal that was not evaluated is NaN; decide
explicitly whether such datasets stay:

```python
ratio = labels["qc_g1_density_ratio"]
keep &= (ratio >= 1.5) | ratio.isna()
```

**Drop cells with a per-cell flag.** Flags are nullable; a missing value
means the flag was not evaluated (an uninformative stratum, or a species
where it is off), so treat it as unset or drop it on purpose:

```python
spill = labels["flag_microglial_spillover"].astype("boolean").fillna(False)
keep &= ~spill.to_numpy(dtype=bool)
```

**Only trusted panels.**

```python
keep &= labels["qc_panel_trust"].astype(str).isin(
    ["validated_real", "validated_simulation"]
)
```

**Reproduce a version-1 result (enforced gate).** Read the strict columns
where the analysis read the plain ones:

```python
strict = labels["ct_subclass_status_strict"].astype(str) == "confident"
strict &= ~labels["exclude_hard_strict"]
names = labels.loc[strict, "ct_subclass_name"]
final = labels["ct_final_name_strict"]
```

There are no strict name columns: the strict pass resolves the same calls
with more levels blocked, so a strict-confident cell's label is its
`ct_<L>_name`.

## With `select_cells`

`merxen.annotation.filtering.select_cells` combines the same conditions into
one boolean mask aligned with the input's index, so an analysis states the
level and the confidence it relies on in one call:

```python
from merxen.annotation.filtering import select_cells

keep = select_cells(
    labels,
    level="subclass",
    within_dataset_support=True,
    exclude_dataset_warnings=["_fail"],
    exclude_flags=["flag_microglial_spillover"],
)
labels[keep]
```

| Argument | Default | Effect |
|---|---|---|
| `level` | required | The level whose `ct_<level>_status` is read. |
| `statuses` | `("confident",)` | The statuses to keep, e.g. `("confident", "low_confidence")` for a sensitivity analysis. |
| `within_dataset_support` | `False` | Keep only cells of datasets whose QC supports `level`: `qc_dataset_max_level` is `level` or deeper in the chain. Needs a chain level, and one species (from the `species` column, or the leaf status column of a clustered `obs`). |
| `exclude_dataset_warnings` | `()` | Drop the cells of datasets with a matching `qc_dataset_warnings` token. A pattern matches a token equal to it, ending with it (`"_fail"`: any fail rule) or starting with it (`"g1_"`: any G1 token). |
| `exclude_flags` | `()` | `flag_*` columns whose true values drop a cell; a missing value keeps it. |
| `exclude_hard` | `True` | Drop `exclude_hard` cells (`exclude_hard_strict` with `strict`). |
| `strict` | `False` | Read the `*_strict` columns: the cells an enforced gate, panel trust and real-data QC would have kept (version 1). |

More calls:

```python
# Confident or low-confidence broad labels, any dataset (sensitivity).
select_cells(labels, level="broad", statuses=("confident", "low_confidence"))

# Exactly the version-1 confident supercluster cells.
select_cells(labels, level="supercluster", strict=True)

# Confident class labels, leaving out any dataset with a G1 token.
select_cells(labels, level="class", exclude_dataset_warnings=["g1_"])
```

On a clustered `obs`, which has no `species` column, `within_dataset_support`
reads the species from the leaf status column. A missing column raises a
`KeyError` that names it.

## Things to know

- **Tokens record reasons, not signals.** A token is a gate reason's code
  plus `_fail` (the reason lowered the advisory level) or `_warn`. Mouse G1
  by default only warns (`mouse_gate.g1_effect = warning`), so a tripped G1
  fail rule appears as `g1_registration_warn`, the same token as a G1 that
  was not evaluated or fell below the warning ratio. `"_fail"` therefore
  does not catch it; filter on `qc_g1_density_ratio` and `qc_g1_shift_um`
  for G1. Real-data QC reasons start with `real_qc_` and end in `_fail` (a
  gate cap, e.g. `real_qc_marker_consistency_fail`) or `_warn` (a warning or
  a withheld level; a per-class check adds its level and class to the
  token).
- **`qc_dataset_max_level` folds in more than the gate level**: a refused or
  missing primary gives `none`, and a level real-data QC withholds stops it
  one level above. A refused panel's labels are kept but untrusted
  (`qc_panel_trust = refused`).
- **`unmappable` cells have no label.** The sample had no reference call to
  resolve (no primary run, or no SEA-AD run for `seaad_subclass`). They are
  not `exclude_hard`, so a filter on `confident` status drops them, and a
  filter on `exclude_hard` alone does not.
- **Combine tables of one species** before `within_dataset_support`; the
  helper refuses a table that holds several. The `qc_*` columns are
  per dataset, so a concatenated table keeps each dataset's own values.
- **Version-1 tables** (written before label table version 2) have no
  `qc_*` or `*_strict` columns, and their plain columns already carry the
  enforced gate. `select_cells` works on them for `level`, `statuses`,
  `exclude_flags` and `exclude_hard`, and raises a `KeyError` for the
  options that need the new columns.
