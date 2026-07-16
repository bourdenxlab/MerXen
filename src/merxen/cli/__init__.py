"""Command-line entry points for MerXen."""

from __future__ import annotations

import logging
import sys

import click

from merxen.cli.run_alignment import (
    align_command,
    alignment_qc_command,
    check_alignment_deps_command,
)
from merxen.cli.run_build_spatialdata import build_spatialdata_command
from merxen.cli.run_clustering_squidpy import clustering_squidpy_command
from merxen.cli.run_comparison import compare_command
from merxen.cli.run_cortical_depth import cortical_depth_command
from merxen.cli.run_enrichment import enrich_command
from merxen.cli.run_mapmycells import mapmycells_command
from merxen.cli.run_mask_image_quantification import (
    mask_image_quantification_command,
)
from merxen.cli.run_qc import qc_command
from merxen.cli.run_segmentation import (
    cellpose_nuclei_segment_command,
    cellpose_segment_command,
    proseg_segment_command,
    segment_command,
)
from merxen.cli.run_spatial_gene_analysis import spatial_gene_analysis_command
from merxen.cli.run_visualization import visualize_command


@click.group(name="merxen")
def main() -> None:
    """MerXen spatial transcriptomics pipeline CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )


main.add_command(build_spatialdata_command)
main.add_command(segment_command)
main.add_command(cellpose_segment_command)
main.add_command(cellpose_nuclei_segment_command)
main.add_command(proseg_segment_command)
main.add_command(enrich_command)
main.add_command(mask_image_quantification_command)
main.add_command(qc_command)
main.add_command(align_command)
main.add_command(alignment_qc_command)
main.add_command(check_alignment_deps_command)
main.add_command(compare_command)
main.add_command(cortical_depth_command)
main.add_command(visualize_command)
main.add_command(spatial_gene_analysis_command)
main.add_command(clustering_squidpy_command)
main.add_command(mapmycells_command)
