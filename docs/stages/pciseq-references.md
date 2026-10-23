# pciSeq reference preparation

`PCISEQ_REFERENCE` builds the linear genes-by-cell-type matrix consumed by
pciSeq. It selects an Allen atlas from the configured species, joins its
anatomical and taxonomy metadata, filters the reference cells, and calculates
expression profiles. A supplied custom matrix is also supported. Each unique
reference configuration is prepared once and shared across matching pipeline
branches.

## Sources and defaults

| Source | Species | Expression input | Default taxonomy level |
|---|---|---|---|
| `wmb` | mouse | Allen whole mouse brain, all 13 raw 10Xv3 shards | `subclass` |
| `whb` | human | Allen whole human brain, raw neuron and non-neuron matrices | `cluster` |
| `seaad` | human | Allen SEA-AD multiregion release, all 10 raw region matrices | `subclass` |
| `custom` | configured species | User-supplied CSV/TSV of linear expression means | Already aggregated |

`auto` resolves to WMB for mouse and WHB for human. Supplying
`pciseq_reference_path` with `source=auto` selects a custom matrix. WHB is the
healthy human whole-brain option; SEA-AD is a separate, explicit choice.
Species/source mismatches are rejected.

WHB uses the native hierarchy `supercluster` → `cluster` → `subcluster`.
It has no level named `subclass`; its intermediate `cluster` level is the
default. WMB supports `class`, `subclass`, `supertype`, and `cluster`;
SEA-AD supports `Class`, `Subclass`, and `Supertype`. Native term-set identifiers
are also accepted. `finest` selects the deepest hierarchical level, excluding
the orthogonal neurotransmitter annotation. Requested levels are validated
against the chosen release rather than silently mapped between atlases.

The implementation pins the official Allen manifest release `20260711` and
its SHA256. Expression and metadata versions are taken from that manifest;
they can differ within a release. WMB intentionally uses the complete 10Xv3
collection rather than mixing protocols. Its broader metadata includes cells
from other protocols; only cells actually present in the selected raw matrices
contribute to profiles.

The source schemas are documented by Allen in the
[WMB tutorial](https://alleninstitute.github.io/abc_atlas_access/notebooks/10x_snRNASeq_tutorial_part_1.html),
[WHB tutorial](https://alleninstitute.github.io/abc_atlas_access/notebooks/WHB-10x_snRNASeq_tutorial_part_1.html),
and [SEA-AD multiregion description](https://github.com/AllenInstitute/abc_atlas_access/blob/main/descriptions/SEA-AD-Multiregion-10X.md).

## Anatomical restriction

Human references require explicitly selected regions by default. For example,
this parameter file uses the WHB frontal cortical regions A44, A45, A46 and A32:

```yaml
pciseq_enabled: true
pciseq_reference_source: whb
pciseq_region_labels: ["Human A44-A45", "Human A46", "Human A32"]
```

Use regions that describe the actual sample. The filter retains cells from
those dissections before calculating profiles and counting support. Types with
insufficient regional support are excluded. A subclass shared across brain
regions keeps its regional expression profile; it is not excluded merely
because it also occurs elsewhere.

WMB uses `region_of_interest_acronym`; WHB uses
`region_of_interest_label`. SEA-AD obtains its region from the library table
joined by `library_label`. SEA-AD's native region labels are `AnG`, `DFC`, `FI`,
`HIP`, `ITG`, `LEC`, `MEC`, `MTG`, `STG`, and `V1C`. For example:

```yaml
pciseq_enabled: true
pciseq_reference_source: seaad
pciseq_region_labels: [DFC]
pciseq_seaad_cohort: all
```

Canonical labels and unambiguous official name/symbol aliases are accepted,
case-insensitively. Unknown or ambiguous regions fail with the available native
labels. Region names are source-specific; there is no automatic cross-atlas
translation. The output manifest includes the native inventory and cell counts.
On a cold cache, acquisition currently obtains all source matrices before
validating these anatomical labels.

Mouse defaults to whole brain when no regions are specified. An explicit mouse
region list enables restriction. **TODO:** connect this selection to future
coronal-section registration to Allen CCFv3, derive regions covered by the slice,
include additional regions within a 100 µm safety margin, then select the
corresponding reference cells/types. This registration is not implemented here.

`pciseq_region_mode=whole_brain` explicitly permits unrestricted references for
either species and requires an empty region list. `required` requires regions
for either species. The default `auto` implements the human/mouse rules above.

## SEA-AD donor selection

The default is **all 84 donors, with donor-balanced profiles**. This includes
the full range of neuropathology and cognitive status; it is not a healthy-only
reference. The release has 42 donors with no dementia and 42 with dementia.
Its AD neuropathological change groups are Not AD (9), Low (12), Intermediate
(21), and High (42). Cognitive status and pathology are distinct attributes.

| `pciseq_seaad_cohort` | Explicit disease-metadata rule | Release-wide donors |
|---|---|---|
| `all` | All donors listed in the released disease table | 84 |
| `control` | Not AD **and** No dementia | 9 |
| `ad` | Intermediate or High AD neuropathological change | 63 |

The selected region and cell type may have fewer represented donors. Support
counts and cohort criteria are recorded rather than assuming every donor
contributes to every profile.

## Expression calculation

For each retained cell, the builder divides raw counts by that cell's total
counts across **all original genes**, then scales to 10,000 by default.
It averages these linear values within each donor and type, then averages
the donor means equally within each type. A donor contributing more cells
therefore does not dominate the reference. Setting donor balance to `false`
uses a cell-weighted arithmetic mean instead.

These are arithmetic means of library-normalized counts, without logarithms,
marker scores, or z-scores. The raw CSR matrices are read in chunks. Ensembl
identifiers are joined to official gene metadata; duplicate symbols are summed.
Library totals are calculated before any optional gene restriction or symbol
collapse. Gene symbols are the default output identifiers; Ensembl output is
configurable. Matching to measured transcript genes remains case-sensitive.

Types need at least 10 contributing cells by default. Empty libraries and
unannotated raw cells are excluded and counted; missing expression for annotated
cells is reported. Invalid raw values, duplicate cell IDs, incompatible schemas,
and non-linear declared normalization fail explicitly. Profiles with no
positive expression are excluded and reported in the support table.

## Storage, integrity, and outputs

Set `MERXEN_REFERENCES_DIR` in the private `.env`, or explicitly provide
`pciseq_reference_root`. Existing `abc_atlas` files are reused per file; newly
acquired official data is placed in `pciseq_atlas` beneath that root. This layer
does not read a legacy SSD location implicitly. Container runs need access to
the reference root and the same local configuration as other pipeline stages.

Official files are checked against manifest sizes and MD5 hashes, assigned
SHA256 fingerprints, and checked for readable CSV/H5AD structure. Verification
is cached against file identity, size, modification time, and the pinned
manifest. Changed files are verified again. Downloads use locks, resumable
partial files, bounded retries, and atomic publication; corrupt writable files
are quarantined. Existing read-only atlas directories are preserved.
Set `pciseq_reference_auto_download=false` to require a complete local cache.
First acquisition and hashing can be substantial: SEA-AD alone has about 76 GB
of raw expression files. Preparation never runs when pciSeq is disabled.

The base environment now declares `h5py` directly. Package versions are unchanged
in the refreshed lock, but its environment checksum changes Nextflow's conda
cache identity. As with other base-environment updates, existing `-resume` tasks
using that environment can run again; preserve published outputs before adopting
the change.

Prepared results are cached under `pciseq_prepared/<content-hash>` in the
reference root. Cache identity includes input content fingerprints, algorithm
version, region/cohort/taxonomy choices, normalization, and support thresholds.
Checksums are validated before reuse; damaged prepared outputs are rebuilt.
The pipeline publishes each bundle under
`<outdir>/pciseq_references/<reference-key>/pciseq_reference_out/`:

- `reference.csv`: genes by retained cell types, directly consumable by pciSeq.
- `class_support.csv`: class IDs, names, contributing cell/donor counts,
  and retention status; custom matrices have no inferred
  single-cell counts.
- `reference_manifest.json`: source release and checksums, effective settings,
  regional inventory, exclusions, normalization, and output checksums.

Each typing branch verifies that the manifest's reference checksum matches the
matrix and carries that manifest into its published results.

## Custom references

Set `pciseq_reference_source=custom` and `pciseq_reference_path` to a CSV/TSV
with genes in the first column and uniquely named cell types in subsequent
columns. Values must be finite, nonnegative linear expression means; `Zero` is
reserved by pciSeq. See [the matrix example](pciseq.md#run-it).

For regional restriction, also supply
`pciseq_reference_custom_metadata_path`, a CSV containing `cell_type` and
`region` columns. One row represents one allowed class/region combination, so
a class can occur in several regions. Only classes associated with a selected
region are retained. Custom region labels match exactly. Alternatively choose
`pciseq_region_mode=whole_brain` to use the entire supplied matrix.

Custom means are preserved without implicit normalization or donor balancing;
those operations require single-cell inputs. The custom matrix and metadata
are fingerprinted and cached with their regional filters.

## Parameters

| Nextflow parameter | Default | Meaning |
|---|---|---|
| `pciseq_reference_source` | `auto` | `auto`, `wmb`, `whb`, `seaad`, or `custom`. |
| `pciseq_reference_root` | `null` | Override the root from `MERXEN_REFERENCES_DIR`. |
| `pciseq_reference_path` | `null` | Custom mean-expression CSV/TSV. |
| `pciseq_region_labels` | `[]` | Source-native regions, list or comma-separated text. |
| `pciseq_region_mode` | `auto` | `auto`, `required`, or `whole_brain`. |
| `pciseq_reference_level` | `auto` | Native taxonomy level/identifier, or `finest`. |
| `pciseq_reference_min_cells` | `10` | Minimum contributing cells per type. |
| `pciseq_reference_normalize_target_sum` | `10000` | Per-cell total for linear count normalization. |
| `pciseq_reference_donor_balance` | `true` | Equal donor weighting within each type. |
| `pciseq_seaad_cohort` | `all` | `all`, `ad`, or `control`; SEA-AD only. |
| `pciseq_reference_auto_download` | `true` | Fetch missing or damaged official assets. |
| `pciseq_reference_verify_checksums` | `true` | Require official MD5 verification as well as structure/size. |
| `pciseq_reference_gene_id_mode` | `symbol` | `symbol` or `ensembl`. |
| `pciseq_reference_custom_metadata_path` | `null` | Custom class/region CSV. |

Samplesheet overrides are available for source, root, path, region labels,
region mode, taxonomy level, SEA-AD cohort, and custom metadata path, as well as
`pciseq_enabled`. Blank cells inherit global values. Quote comma-separated
region lists inside CSV cells.

## Direct preparation

`merxen prepare-pciseq-reference --config reference_config.json` runs the same
builder outside Nextflow. Example JSON for mouse whole brain:

```json
{
  "species": "mouse",
  "source": "auto",
  "output_dir": "outputs/reference"
}
```

JSON uses `PciSeqReferenceConfig` fields: `level`, `min_cells_per_type`,
`normalization_target_sum`, `donor_balance`, `auto_download`, `verify_checksums`,
`custom_metadata_path`, and the other names without the `pciseq_` prefix.
It also supports `chunk_rows` (default 2048) and optional `gene_names`.
Use the resulting matrix and manifest as `reference_path` and
`reference_manifest_path` for the direct typing CLI.
