"""Write the MerXen samplesheet of the dentate-gyrus lifespan Xenium dataset.

One ``analysis_mode=xenium`` row per section, named ``PUBXDG_<label>`` (the
pair ids of ``raw/listing.json``). Paths are resolved at run time from
``$MERXEN_XENIUM_PUBLIC_ROOT`` (``.env`` via ``merxen.local_config``) and
written into ``workflows/samplesheet.public_dg_lifespan.csv``, which is
gitignored (``workflows/*.csv``) because it holds host paths.

The row options mirror the latest Atera hippocampus rows as far as a Xenium
row allows:

- ``segmentation_source=cellpose_proseg`` (the only Xenium route) with
  ``xenium_channels=DAPI``: the deposit has only the DAPI z-stack, and a
  custom-model arm needs every requested channel to exist.
- ``analysis_segmentation=all``, ``start_stage=build_spatialdata``,
  ``stop_stage=mender``, ``mender_enabled=true``.
- ``spatial_gene_analysis_transcript_analysis_enabled=false``: otherwise a
  Xenium row needs pial and tissue-edge GeoJSON files.
- ``mecr_enabled=false`` as in the AT-11 Atera run (``--mecr`` turns it on).
- Annotation: human ``map_first`` accepts only ``frontal_cortex`` on Xenium
  rows (the hippocampal profile is Atera-only), so the default is
  ``clustering_squidpy_mode=legacy`` with ``anatomical_region=hippocampus``
  recorded; ``--annotation map_first_frontal`` leaves the region blank (the
  run's frontal-cortex default) and runs map-first anyway.
- ``xenium_min_qv=20`` and ``xenium_voxel_layers=2`` (the defaults, written
  out).
- ``xenium_spatialdata_path`` is left blank: the source zarr is published
  under the run's outdir. ``--cache-source-zarr`` fills it with
  ``.../dg_lifespan/spatialdata/<pair_id>_xenium_source.zarr`` so other runs
  reuse the build, but the pipeline also takes that column as the *latest*
  (enriched) zarr when a run starts after ``enrich``, so a sheet with the
  column filled must not be used for such a restart.

Launch with ``--species human``; with the apptainer profile the public root
must be in ``apptainer_bind_paths``.

Usage::

    python scripts/public_data/xenium_hippocampus/make_samplesheet_dg_lifespan.py
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from merxen import local_config

logger = logging.getLogger("make_samplesheet_dg_lifespan")

DATASET_KEY: Final = "dg_lifespan"
DATASET_DIR: Final = "xenium_prime_5k_proto_human_dentate_gyrus_ff"
REPO_ROOT: Final = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT: Final = REPO_ROOT / "workflows" / "samplesheet.public_dg_lifespan.csv"
PROVENANCE_FILE: Final = "merxen_provenance.json"
COLUMNS: Final = (
    "pair_id",
    "analysis_mode",
    "enable_alignment",
    "segmentation_source",
    "analysis_segmentation",
    "start_stage",
    "stop_stage",
    "mender_enabled",
    "mecr_enabled",
    "spatial_gene_analysis_transcript_analysis_enabled",
    "clustering_squidpy_mode",
    "anatomical_region",
    "xenium_dir",
    "xenium_spatialdata_path",
    "xenium_channels",
    "xenium_min_qv",
    "xenium_voxel_layers",
)
ANNOTATION_MODES: Final = ("legacy", "map_first_frontal")
#: spatialdata-io 0.6 names the single channel of an XOA 2/3
#: ``morphology_focus/`` folder ``DAPI`` whatever its OME-XML says, so this is
#: the only channel name the built zarr can carry.
XENIUM_CHANNELS: Final = "DAPI"


class SamplesheetError(RuntimeError):
    """Raised when no sections can be found."""


@dataclass
class SectionRow:
    """What a row needs to know about one section.

    Attributes:
        pair_id: MerXen pair id.
        accession: GSM accession (logged, not written).
        is_prepared: Whether ``prepare_dg_lifespan.py`` finished the folder.
        label: Short section label (``--sections`` matches it too).
    """

    pair_id: str
    accession: str
    is_prepared: bool
    label: str = ""


def dataset_dir(public_root: Path) -> Path:
    """Return ``<public root>/xenium_prime_5k_proto_human_dentate_gyrus_ff``."""
    return public_root / DATASET_DIR


def discover_sections(root: Path, listing: Path | None = None) -> list[SectionRow]:
    """Return the sections from ``raw/listing.json`` and the prepared folders.

    Args:
        root: The dataset directory.
        listing: Listing file to read instead of ``<root>/raw/listing.json``
            (e.g. one written by ``fetch_dg_lifespan.py --write-listing``
            before anything is downloaded).

    Returns:
        One entry per listed section (listing order), plus prepared folders
        that the listing lacks.

    Raises:
        SamplesheetError: If neither source names a section.
    """
    rows: dict[str, SectionRow] = {}
    listing = listing or root / "raw" / "listing.json"
    if listing.is_file():
        data = json.loads(listing.read_text(encoding="utf-8"))
        for section in data.get("sections", []):
            rows[section["pair_id"]] = SectionRow(
                pair_id=section["pair_id"],
                accession=section["accession"],
                is_prepared=False,
                label=str(section.get("label") or ""),
            )
    xenium_dir = root / "xenium"
    if xenium_dir.is_dir():
        for folder in sorted(xenium_dir.iterdir()):
            stamp = folder / PROVENANCE_FILE
            if not stamp.is_file():
                continue
            record = json.loads(stamp.read_text(encoding="utf-8"))
            pair_id = str(record.get("pair_id") or folder.name)
            row = rows.setdefault(
                pair_id,
                SectionRow(pair_id, str(record.get("geo_accession", "")), False),
            )
            row.is_prepared = True
            row.label = row.label or str(record.get("label") or "")
    if not rows:
        raise SamplesheetError(
            f"no sections under {root}: run fetch_dg_lifespan.py --metadata-only first"
        )
    return list(rows.values())


def build_rows(
    sections: Sequence[SectionRow],
    root: Path,
    *,
    annotation: str = "legacy",
    mecr_enabled: bool = False,
    cache_source_zarr: bool = False,
) -> list[dict[str, str]]:
    """Return one samplesheet row per section.

    Args:
        sections: The sections.
        root: The dataset directory (absolute paths are derived from it).
        annotation: ``legacy`` or ``map_first_frontal`` (see module doc).
        mecr_enabled: Value of ``mecr_enabled``.
        cache_source_zarr: Fill ``xenium_spatialdata_path`` (see module doc).

    Returns:
        Rows keyed by :data:`COLUMNS`.
    """
    if annotation not in ANNOTATION_MODES:
        raise SamplesheetError(f"unknown annotation mode {annotation!r}")
    is_legacy = annotation == "legacy"
    rows = []
    for section in sections:
        rows.append(
            {
                "pair_id": section.pair_id,
                "analysis_mode": "xenium",
                "enable_alignment": "false",
                "segmentation_source": "cellpose_proseg",
                "analysis_segmentation": "all",
                "start_stage": "build_spatialdata",
                "stop_stage": "mender",
                "mender_enabled": "true",
                "mecr_enabled": "true" if mecr_enabled else "false",
                "spatial_gene_analysis_transcript_analysis_enabled": "false",
                "clustering_squidpy_mode": "legacy" if is_legacy else "map_first",
                "anatomical_region": "hippocampus" if is_legacy else "",
                "xenium_dir": str(root / "xenium" / section.pair_id),
                "xenium_spatialdata_path": (
                    str(root / "spatialdata" / f"{section.pair_id}_xenium_source.zarr")
                    if cache_source_zarr
                    else ""
                ),
                "xenium_channels": XENIUM_CHANNELS,
                "xenium_min_qv": "20",
                "xenium_voxel_layers": "2",
            }
        )
    return rows


def render_csv(rows: Sequence[dict[str, str]]) -> str:
    """Return the CSV text of the rows (header from :data:`COLUMNS`)."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(COLUMNS), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    local_config.add_location_argument(
        parser,
        "--public-root",
        "MERXEN_XENIUM_PUBLIC_ROOT",
        "Root of the public validation datasets.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--listing",
        type=Path,
        default=None,
        help="Listing JSON to use instead of <dataset>/raw/listing.json.",
    )
    parser.add_argument("--annotation", choices=ANNOTATION_MODES, default="legacy")
    parser.add_argument("--mecr", action="store_true", help="Write mecr_enabled=true.")
    parser.add_argument(
        "--cache-source-zarr",
        action="store_true",
        help=(
            "Fill xenium_spatialdata_path so the source zarr is built once under "
            "the dataset folder. Do not restart such a sheet after enrich: the "
            "pipeline then reads that column as the latest (enriched) zarr."
        ),
    )
    parser.add_argument(
        "--sections",
        default="",
        help="Comma-separated pair ids, GSMs or labels (default: all).",
    )
    parser.add_argument(
        "--require-prepared",
        action="store_true",
        help="Fail when a selected section has no prepared folder yet.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Write the samplesheet.

    Args:
        argv: Command-line arguments (``sys.argv[1:]`` when ``None``).

    Returns:
        The process exit code.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    root = dataset_dir(args.public_root)
    sections = discover_sections(root, args.listing)
    wanted = {w.strip().upper() for w in args.sections.split(",") if w.strip()}
    if wanted:
        sections = [
            s
            for s in sections
            if {s.pair_id.upper(), s.accession.upper(), s.label.upper()} & wanted
        ]
        if not sections:
            raise SamplesheetError(f"no section matches --sections {args.sections}")
    unprepared = [s.pair_id for s in sections if not s.is_prepared]
    if unprepared:
        message = "not prepared yet (run prepare_dg_lifespan.py): " + ", ".join(
            unprepared
        )
        if args.require_prepared:
            logger.error(message)
            return 1
        logger.warning(message)
    rows = build_rows(
        sections,
        root,
        annotation=args.annotation,
        mecr_enabled=args.mecr,
        cache_source_zarr=args.cache_source_zarr,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_csv(rows), encoding="utf-8")
    logger.info("Wrote %d rows to %s", len(rows), args.output)
    if args.annotation == "map_first_frontal":
        logger.warning(
            "map_first on these rows uses the frontal-cortex references on "
            "hippocampal tissue; expect frontal-cortex plausibility flags."
        )
    if args.cache_source_zarr:
        logger.warning(
            "xenium_spatialdata_path is filled: do not restart this sheet with a "
            "start_stage after enrich (that column then names the latest zarr)."
        )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SamplesheetError as error:
        logger.error("%s", error)
        sys.exit(1)
