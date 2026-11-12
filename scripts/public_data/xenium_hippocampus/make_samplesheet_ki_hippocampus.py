#!/usr/bin/env python
"""Write the MerXen samplesheet of the KI hippocampus Xenium sections.

One ``analysis_mode=xenium`` row per section, set up to run as closely as a
Xenium row can to the Atera hippocampus rows (Cellpose + ProSeg, every
segmentation branch, ``build_spatialdata`` through MENDER, MECR off as in
AT-11). The sheet is written to
``workflows/samplesheet.public_ki_hippocampus.csv``, which is gitignored
(``workflows/*.csv``) because it holds host paths.

Choices made for every row, with their reasons:

- ``xenium_channels=DAPI``: XOA 1.4 has DAPI only (no 18S / boundary stains).
  spatialdata-io names that single pre-2.0 channel ``0``; MerXen's Xenium build
  names it ``DAPI``.
- ``spatial_gene_analysis_transcript_analysis_enabled=false``: it would need
  pial and tissue-edge GeoJSON, which do not exist for these sections.
- ``clustering_squidpy_mode=legacy`` (default): human ``map_first`` on a Xenium
  row has validated references for ``frontal_cortex`` only; the hippocampal
  profile is Atera-only. ``--clustering-mode map_first`` leaves
  ``anatomical_region`` blank instead (frontal-cortex references).
- ``mecr_enabled=false`` (default) as in the AT-11 Atera run.
- ``xenium_spatialdata_path`` is left blank: the source zarr is published
  under the run's outdir. ``--cache-source-zarr`` fills it with
  ``.../xenium_v1_human_hippocampus_ff_ki/spatialdata/<pair_id>_xenium.zarr``
  so other runs reuse the build, but the pipeline also takes that column as
  the *latest* (enriched) zarr when a run starts after ``enrich``, so a sheet
  with the column filled must not be used for such a restart.

Paths come from ``$MERXEN_XENIUM_PUBLIC_ROOT`` (``--public-root``).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import logging
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Final

from merxen import local_config

logger = logging.getLogger(__name__)

REPO_ROOT: Final = Path(__file__).resolve().parents[3]
SAMPLESHEET_NAME: Final = "samplesheet.public_ki_hippocampus.csv"
COLUMNS: Final[tuple[str, ...]] = (
    "pair_id",
    "analysis_mode",
    "enable_alignment",
    "segmentation_source",
    "analysis_segmentation",
    "start_stage",
    "stop_stage",
    "xenium_dir",
    "xenium_spatialdata_path",
    "xenium_channels",
    "xenium_min_qv",
    "xenium_voxel_layers",
    "mender_enabled",
    "mecr_enabled",
    "spatial_gene_analysis_transcript_analysis_enabled",
    "clustering_squidpy_mode",
    "anatomical_region",
)


def _load_sibling(name: str) -> ModuleType:
    key = f"_ki_hippocampus_{name}"
    if key in sys.modules:
        return sys.modules[key]
    path = Path(__file__).resolve().with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(key, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


fetch = _load_sibling("fetch_ki_hippocampus")
prepare = _load_sibling("prepare_ki_hippocampus")


class SamplesheetError(RuntimeError):
    """The samplesheet could not be written safely."""


@dataclass(frozen=True)
class RowOptions:
    """Settings shared by every row.

    Attributes:
        clustering_mode: ``legacy`` or ``map_first``.
        mecr_enabled: Row-level MECR switch.
        min_qv: ``xenium_min_qv``.
        voxel_layers: ``xenium_voxel_layers``.
        cache_source_zarr: Fill ``xenium_spatialdata_path`` (see module doc).
    """

    clustering_mode: str = "legacy"
    mecr_enabled: bool = False
    min_qv: int = 20
    voxel_layers: int = 2
    cache_source_zarr: bool = False


def discover_sections(public_root: Path, sections: Sequence[str] | None) -> list[str]:
    """Return the section labels to write rows for.

    Order of sources: ``sections`` when given, else the fetch manifest, else
    the staged ``xenium/PUBXHPC_*`` folders.

    Args:
        public_root: Public-data root.
        sections: Explicit sections (``S1``, ``1`` or ``KI_HC_S1``).

    Returns:
        Labels in section order.

    Raises:
        SamplesheetError: If no section can be found.
    """
    if sections:
        try:
            labels = [fetch.normalise_section(value) for value in sections]
        except fetch.FetchError as exc:
            raise SamplesheetError(str(exc)) from exc
    else:
        manifest = fetch.metadata_dir(public_root) / fetch.MANIFEST_NAME
        if manifest.is_file():
            labels = [item.section for item in fetch.read_manifest(manifest)]
        else:
            staged = fetch.dataset_root(public_root) / "xenium"
            labels = []
            for path in sorted(staged.glob(f"{prepare.PAIR_ID_PREFIX}*")):
                suffix = path.name[len(prepare.PAIR_ID_PREFIX) :]
                try:
                    label = fetch.normalise_section(suffix)
                except fetch.FetchError:
                    label = None
                if not path.is_dir() or label != suffix:
                    logger.warning("Skipping %s: not a staged section folder", path)
                    continue
                labels.append(label)
    if not labels:
        raise SamplesheetError(
            "no sections found: run fetch_ki_hippocampus.py --metadata-only, "
            "or pass --sections"
        )
    prefix = len(fetch.SECTION_LABEL_PREFIX)
    return sorted(set(labels), key=lambda label: int(label[prefix:]))


def build_row(public_root: Path, section: str, options: RowOptions) -> dict[str, str]:
    """Return the samplesheet row of one section.

    Args:
        public_root: Public-data root.
        section: Section label.
        options: Shared settings.

    Returns:
        Column -> value.
    """
    is_legacy = options.clustering_mode == "legacy"
    return {
        "pair_id": prepare.pair_id_for(section),
        "analysis_mode": "xenium",
        "enable_alignment": "false",
        "segmentation_source": "cellpose_proseg",
        "analysis_segmentation": "all",
        "start_stage": "build_spatialdata",
        "stop_stage": "mender",
        "xenium_dir": str(prepare.xenium_dir(public_root, section)),
        "xenium_spatialdata_path": (
            str(prepare.spatialdata_path(public_root, section))
            if options.cache_source_zarr
            else ""
        ),
        "xenium_channels": "DAPI",
        "xenium_min_qv": str(options.min_qv),
        "xenium_voxel_layers": str(options.voxel_layers),
        "mender_enabled": "true",
        "mecr_enabled": "true" if options.mecr_enabled else "false",
        "spatial_gene_analysis_transcript_analysis_enabled": "false",
        "clustering_squidpy_mode": options.clustering_mode,
        # Read only by map_first; legacy ignores it, so it records the tissue.
        "anatomical_region": "hippocampus" if is_legacy else "",
    }


def is_git_ignored(path: Path, repo_root: Path = REPO_ROOT) -> bool | None:
    """Ask git whether ``path`` is ignored.

    Returns:
        True or False, or None when git cannot answer (no git, no repository).
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "check-ignore", "-q", str(path)],
            check=False,
            capture_output=True,
        )
    except OSError:
        return None
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    return None


def write_samplesheet(path: Path, rows: Sequence[dict[str, str]]) -> None:
    """Write rows as a CSV with the fixed column order.

    Raises:
        SamplesheetError: If git reports the path as tracked-able (not ignored).
    """
    if is_git_ignored(path) is False:
        raise SamplesheetError(
            f"{path} is not gitignored; it would hold host paths. Write it under "
            "workflows/ (workflows/*.csv is ignored) or pass another ignored path."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    local_config.add_location_argument(
        parser,
        "--public-root",
        "MERXEN_XENIUM_PUBLIC_ROOT",
        "Public-data root holding xenium_v1_human_hippocampus_ff_ki/",
    )
    parser.add_argument("--sections", nargs="+", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "workflows",
        help="Folder for the samplesheet (default: workflows/).",
    )
    parser.add_argument(
        "--clustering-mode", choices=("legacy", "map_first"), default="legacy"
    )
    parser.add_argument("--mecr", choices=("true", "false"), default="false")
    parser.add_argument("--min-qv", type=int, default=20)
    parser.add_argument("--voxel-layers", type=int, default=2)
    parser.add_argument(
        "--cache-source-zarr",
        action="store_true",
        help=(
            "Fill xenium_spatialdata_path so the source zarr is built once under "
            "the dataset folder. Do not restart such a sheet after enrich: the "
            "pipeline then reads that column as the latest (enriched) zarr."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the samplesheet generator.

    Args:
        argv: Command-line arguments (``sys.argv[1:]`` when ``None``).

    Returns:
        The process exit code.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    options = RowOptions(
        clustering_mode=args.clustering_mode,
        mecr_enabled=args.mecr == "true",
        min_qv=args.min_qv,
        voxel_layers=args.voxel_layers,
        cache_source_zarr=args.cache_source_zarr,
    )
    try:
        sections = discover_sections(args.public_root, args.sections)
        sheet = args.output_dir / SAMPLESHEET_NAME
        write_samplesheet(
            sheet, [build_row(args.public_root, s, options) for s in sections]
        )
    except (SamplesheetError, fetch.FetchError) as error:
        logger.error("%s", error)
        return 2
    for section in sections:
        if not (
            prepare.xenium_dir(args.public_root, section) / "experiment.xenium"
        ).is_file():
            logger.warning(
                "%s: not staged yet (run prepare_ki_hippocampus.py)", section
            )
    logger.info("Wrote %s (%d rows).", sheet, len(sections))
    logger.info(
        "Run (add your -profile; --species human as for the Atera rows):\n"
        "  nextflow run workflows/main.nf --species human "
        "--samplesheet %s --outdir <outdir>\n"
        "With -profile apptainer, $MERXEN_XENIUM_PUBLIC_ROOT must be in "
        "apptainer_bind_paths.",
        sheet,
    )
    if args.cache_source_zarr:
        logger.warning(
            "xenium_spatialdata_path is filled: do not restart this sheet with a "
            "start_stage after enrich (that column then names the latest zarr)."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
