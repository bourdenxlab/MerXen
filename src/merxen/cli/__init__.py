"""Command-line entry points for MerXen."""

from __future__ import annotations

import logging
import sys

import click

from merxen.cli.run_alignment import (
    align_command,
    alignment_qc_command,
    check_alignment_deps_command,
    materialize_alignment_command,
)
from merxen.cli.run_analysis_gene_set import analysis_gene_set_command
from merxen.cli.run_analysis_layer_validation import (
    validate_analysis_layer_command,
)
from merxen.cli.run_annotation import (
    annotate_command,
    annotate_resolve_command,
    annotation_panel_command,
    annotation_reference_prep_command,
    annotation_store_group,
)
from merxen.cli.run_annotation_panels import (
    annotation_panel_fetch_command,
    annotation_panel_simulate_command,
)
from merxen.cli.run_annotation_report import annotation_report_command
from merxen.cli.run_atera_proseg import (
    atera_proseg_plan_command,
    atera_proseg_stitch_command,
    atera_proseg_tile_command,
)
from merxen.cli.run_build_spatialdata import build_spatialdata_command
from merxen.cli.run_cellpose_hitl import (
    cellpose_hitl_export_command,
    cellpose_hitl_import_command,
    cellpose_hitl_prepare_rois_command,
    cellpose_hitl_seed_command,
    cellpose_hitl_train_command,
)
from merxen.cli.run_clustering_squidpy import clustering_squidpy_command
from merxen.cli.run_comparison import compare_command
from merxen.cli.run_convert_czi import convert_czi_command
from merxen.cli.run_cortical_depth import cortical_depth_command
from merxen.cli.run_distance_from_object import (
    distance_from_object_cohort_command,
    distance_from_object_command,
)
from merxen.cli.run_enrichment import enrich_command
from merxen.cli.run_image_registration import register_images_command
from merxen.cli.run_manual_registration_prepare import (
    manual_registration_prepare_command,
)
from merxen.cli.run_manual_registration_register import (
    manual_registration_compare_command,
    manual_registration_fit_command,
    manual_registration_register_command,
)
from merxen.cli.run_mapmycells import mapmycells_command
from merxen.cli.run_mask_image_quantification import (
    mask_image_quantification_command,
)
from merxen.cli.run_mecr import mecr_command, mecr_reference_command
from merxen.cli.run_native_latest import native_latest_command
from merxen.cli.run_pciseq import pciseq_command, publish_pciseq_viewer_command
from merxen.cli.run_pciseq_reference import prepare_pciseq_reference_command
from merxen.cli.run_qc import qc_command
from merxen.cli.run_segmentation import (
    cellpose_nuclei_segment_command,
    cellpose_segment_command,
    proseg_segment_command,
    segment_command,
)
from merxen.cli.run_spatial_gene_analysis import spatial_gene_analysis_command
from merxen.cli.run_transcript_density import (
    atera_transcript_density_command,
    embed_transcript_density_command,
)
from merxen.cli.run_viewer_cache import build_viewer_caches_command
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
main.add_command(atera_transcript_density_command)
main.add_command(embed_transcript_density_command)
main.add_command(segment_command)
main.add_command(cellpose_segment_command)
main.add_command(cellpose_hitl_export_command)
main.add_command(cellpose_hitl_prepare_rois_command)
main.add_command(cellpose_hitl_import_command)
main.add_command(cellpose_hitl_train_command)
main.add_command(cellpose_hitl_seed_command)
main.add_command(cellpose_nuclei_segment_command)
main.add_command(proseg_segment_command)
main.add_command(native_latest_command)
main.add_command(atera_proseg_plan_command)
main.add_command(atera_proseg_tile_command)
main.add_command(atera_proseg_stitch_command)
main.add_command(enrich_command)
main.add_command(register_images_command)
main.add_command(build_viewer_caches_command)
main.add_command(mask_image_quantification_command)
main.add_command(validate_analysis_layer_command)
main.add_command(qc_command)
main.add_command(align_command)
main.add_command(materialize_alignment_command)
main.add_command(alignment_qc_command)
main.add_command(check_alignment_deps_command)
main.add_command(compare_command)
main.add_command(cortical_depth_command)
main.add_command(distance_from_object_command)
main.add_command(distance_from_object_cohort_command)
main.add_command(visualize_command)
main.add_command(spatial_gene_analysis_command)
main.add_command(mecr_reference_command)
main.add_command(mecr_command)
main.add_command(clustering_squidpy_command)
main.add_command(mapmycells_command)
main.add_command(pciseq_command)
main.add_command(publish_pciseq_viewer_command)
main.add_command(prepare_pciseq_reference_command)
main.add_command(convert_czi_command)
main.add_command(manual_registration_prepare_command)
main.add_command(manual_registration_register_command)
main.add_command(manual_registration_fit_command)
main.add_command(manual_registration_compare_command)
main.add_command(analysis_gene_set_command)
main.add_command(annotation_panel_command)
main.add_command(annotation_reference_prep_command)
main.add_command(annotation_store_group)
main.add_command(annotation_panel_fetch_command)
main.add_command(annotation_panel_simulate_command)
main.add_command(annotate_command)
main.add_command(annotate_resolve_command)
main.add_command(annotation_report_command)
