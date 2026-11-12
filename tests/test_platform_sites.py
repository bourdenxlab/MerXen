"""Pins on the code that still hard-codes platform names.

The platform registry (``src/merxen/platforms.py`` and its twin
``workflows/lib/PlatformSpec.groovy``) is the one place that names a platform's
original shape layers and native image. Some sites still spell them out: ordered
priority lists, display code and dispatch that already raises for an unknown
platform (the ``qc.nf`` script has looked its reference shape up in the
registry since AT-6). Each category below pins those sites by line text (not
line number, so the pin survives unrelated edits) together with the reason
they are not converted yet.

A new matching line fails the test. Look the value up in the registry instead,
or, when the line must name the platform, add it to the pin with a reason. A
line that disappeared fails too, so the pin stays an exact inventory.

The nucleus categories (R6 in the platform-support plan) list every consumer of
vendor nucleus layers and nucleus masks: each one must be routed through the
cell/nucleus identity mapping before a platform whose nucleus ids are not cell
ids can be supported. ``nuclei_mask_consumers`` also lists the files that read
or write MerXen's own Cellpose nuclei layer (``cellpose_nuclei``), so a new
nucleus-to-cell join on that layer is a reviewed change too.

``built_shape_names`` catches shape names assembled from a prefix
(``f"{prefix}_nucleus"``, ``platform.lower() + "_cell_boundaries"``), which the
literal ``shape_names`` pattern cannot see. ``platform_equality`` covers every
spelling of a platform dispatch: ``==``/``!=`` in either case, membership in a
literal tuple, list or set, ``.equals()``, ``case`` labels and ``startswith``.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REGISTRY_FILES = frozenset(
    {"src/merxen/platforms.py", "workflows/lib/PlatformSpec.groovy"}
)
_SCANNED_GLOBS: tuple[str, ...] = (
    "src/merxen/**/*.py",
    "workflows/main.nf",
    "workflows/modules/*.nf",
    "workflows/lib/*.groovy",
    "scripts/**/*.py",
    "workflows/acceptance/*.nf",
)


@dataclass(frozen=True)
class SiteCategory:
    """A family of hard-coded platform values.

    Attributes:
        name: Identifier used in the pins and failure messages.
        pattern: Regular expression a line matches.
        file_level: Pin only the files that match, not their lines.
    """

    name: str
    pattern: str
    file_level: bool = False


CATEGORIES: tuple[SiteCategory, ...] = (
    SiteCategory(
        "shape_names",
        r"\b(merscope|xenium)_(cell_boundaries|nucleus)(_aligned_nonrigid)?\b"
        r"|lower\(\)\}_cell_boundaries",
    ),
    SiteCategory(
        "built_shape_names",
        r"[\"']_(cell_boundaries|nucleus)[\"']|\}_(cell_boundaries|nucleus)\b",
    ),
    SiteCategory("vendor_nucleus_names", r"\b(xenium_nucleus|nucleus_boundaries)\b"),
    SiteCategory(
        "nucleus_joins",
        r"_choose_nucleus_shape_key|nucleus_area|\"nucleus\" (not )?in str\(key\)",
    ),
    SiteCategory(
        "nuclei_mask_consumers", r"nuclei_mask|cellpose_nuclei|nuclei_labels", True
    ),
    SiteCategory(
        "platform_equality",
        r"(?i)(==|!=)\s*[\"'](MERSCOPE|XENIUM)[\"']"
        r"|[\"'](MERSCOPE|XENIUM)[\"']\s*(==|!=)"
        r"|\bin\s*[(\[{][^)\]}]*[\"'](MERSCOPE|XENIUM)[\"']"
        r"|[\"'](MERSCOPE|XENIUM)[\"']\.equals(IgnoreCase)?\("
        r"|\.equals(IgnoreCase)?\(\s*[\"'](MERSCOPE|XENIUM)[\"']"
        r"|\bcase\s+[\"'](MERSCOPE|XENIUM)[\"']"
        r"|startswith\(\s*[\"'](MERSCOPE|XENIUM)[\"']",
    ),
    SiteCategory("native_image_names", r"morphology_focus|MERSCOPE_z_projection"),
)

# Category -> file -> reason the file keeps its sites (every pinned file needs one).
REASONS: dict[str, dict[str, str]] = {
    "shape_names": {
        "src/merxen/alignment/legacy_features.py": (
            "ordered shape priority list; registry order could reorder it (AT-6)"
        ),
        "src/merxen/analysis/clustering_squidpy.py": (
            "ordered shape priority lists, aligned variants first (AT-6)"
        ),
        "src/merxen/analysis/pciseq.py": (
            "ordered priority list, XENIUM listed first (AT-6)"
        ),
        "src/merxen/visualization/sanity_plots.py": (
            "display styles and draw order of the original layers"
        ),
    },
    "built_shape_names": {
        "src/merxen/masks.py": (
            "id namespace of the original nucleus mask branch, not a shape key"
        ),
    },
    "vendor_nucleus_names": {
        "scripts/public_data/xenium_hippocampus/fetch_dg_lifespan.py": (
            "public-data intake shim: writes and checks the vendor Xenium"
            " output layout (file names), not registry layers"
        ),
        "scripts/public_data/xenium_hippocampus/prepare_dg_lifespan.py": (
            "public-data intake shim: writes and checks the vendor Xenium"
            " output layout (file names), not registry layers"
        ),
        "scripts/public_data/xenium_hippocampus/prepare_ki_hippocampus.py": (
            "public-data intake shim: writes and checks the vendor Xenium"
            " output layout (file names), not registry layers"
        ),
        "src/merxen/analysis/clustering_squidpy.py": (
            "nucleus shape priority list for platforms whose nucleus ids are cell "
            "ids; ATERA uses the registered nucleus layer instead (AT-6)"
        ),
        "src/merxen/config.py": "Xenium and Atera reader options (build config fields)",
        "src/merxen/enrichment/enrich.py": (
            "raw Xenium reader inputs and source layer names; nucleus consumer for AT-6"
        ),
        "src/merxen/io/atera_reader.py": (
            "Atera reader produces per-nucleus shapes keyed by nucleus id with "
            "cell_instance_id (D7)"
        ),
        "src/merxen/io/builders/atera.py": (
            "Atera builder writes the per-nucleus source shapes element (D7); "
            "registered as atera_nucleus only in AT-6"
        ),
        "src/merxen/io/builders/xenium.py": "Xenium reader option",
        "src/merxen/masks.py": (
            "legacy alias of the Xenium original_nucleus branch; the branch list "
            "is built from the registry since AT-6"
        ),
        "src/merxen/segmentation/native.py": (
            "NATIVE_LATEST renames the source nucleus shapes to the registry's "
            "atera_nucleus; the source name must not reach the latest store (AT-6)"
        ),
        "src/merxen/segmentation/pipeline.py": "Xenium reader option",
    },
    "nucleus_joins": {
        "scripts/public_data/xenium_hippocampus/prepare_dg_lifespan.py": (
            "public-data intake shim: writes and checks the vendor Xenium"
            " output layout (file names), not registry layers"
        ),
        "scripts/acceptance/atera_build.py": (
            "Atera acceptance compares the table's nucleus_area with "
            "cell_summary and skips the vendor float32 copy (AT-5)"
        ),
        "scripts/acceptance/make_atera_crop.py": (
            "Atera crop keeps a cell only when its largest nucleus "
            "(cell_summary.nucleus_area) lies whole in the window (AT-5)"
        ),
        "src/merxen/analysis/clustering_squidpy.py": (
            "nucleus area metrics: by id where nucleus ids are cell ids, else "
            "aggregated per cell through cell_instance_id (AT-6)"
        ),
        "src/merxen/io/atera_reader.py": (
            "Atera table reads the largest-nucleus area and sums all nuclei per cell "
            "through polygon_sets/0/cell_index (D7)"
        ),
        "src/merxen/io/builders/atera.py": (
            "Atera builder passes the nucleus areas counted during the masks/0 "
            "export to the table (D7)"
        ),
        "src/merxen/io/spatialdata_io.py": "cell layer selection excludes nuclei",
        "src/merxen/visualization/sanity_plots.py": (
            "display layer selection excludes nuclei"
        ),
    },
    "nuclei_mask_consumers": {
        "src/merxen/cli/__init__.py": "registers the nuclei segmentation command",
        "src/merxen/segmentation/hitl/seed_inputs.py": (
            "copies completed stock Cellpose nuclei artifacts into a fresh arm; "
            "does not join vendor nucleus ids to cells"
        ),
        "workflows/lib/CellposeHitlSettings.groovy": (
            "requires/stages existing stock Cellpose nuclei for late starts; "
            "Atera retains vendor nuclei and never enters this copy path"
        ),
        "src/merxen/cli/run_segmentation.py": "runs the nuclei segmentation",
        "src/merxen/config.py": "config fields of the nuclei mask",
        "src/merxen/enrichment/enrich.py": "builds the cellpose_nuclei layer (AT-6)",
        "src/merxen/masks.py": "mask-branch discovery of cellpose_nuclei (AT-6)",
        "src/merxen/registration_quality.py": "reads the nuclei mask (AT-6)",
        "src/merxen/segmentation/pipeline.py": "writes the nuclei mask",
        "src/merxen/viewer_cache/build.py": "default viewer layer list",
        "workflows/main.nf": "routes the nuclei mask between stages",
        "workflows/modules/enrichment.nf": "passes the nuclei mask to enrichment",
        "workflows/modules/segmentation.nf": (
            "produces and forwards the nuclei mask (process script text is pinned)"
        ),
        "scripts/acceptance/segmentation_comparison.py": (
            "RCA acceptance script: reads the cellpose_nuclei shapes of paired "
            "MERSCOPE/XENIUM stores (never run on ATERA)"
        ),
    },
    "platform_equality": {
        "src/merxen/segmentation/hitl/import_annotations.py": (
            "managed training pilot supports the same Atera/Xenium native ROI "
            "contract as the exporter; MERSCOPE annotation support is deferred"
        ),
        "workflows/lib/CellposeHitlSettings.groovy": (
            "training is restricted to the two platforms supported by the "
            "current annotation exporter and CellposeTrainingConfig"
        ),
        "src/merxen/segmentation/hitl/regions.py": (
            "initial native-pixel ROI exchange supports ATERA and XENIUM only; "
            "MERSCOPE affine ROI support is a later extension"
        ),
        "scripts/cleanup_assignment_tables_and_outputs.py": (
            "analysis_mode map, kept in step by test_analysis_mode_parity"
        ),
        "src/merxen/alignment/dapi.py": "pairing role: MERSCOPE is the moving image",
        "src/merxen/alignment/frames.py": (
            "image preference list with a Xenium MIP fallback (pairing roles)"
        ),
        "src/merxen/alignment/manifest.py": "pairing role: moving platform",
        "src/merxen/alignment/materialize.py": "pairing roles (XENIUM fixed)",
        "src/merxen/alignment/pipeline.py": "pairing roles (XENIUM fixed)",
        "src/merxen/alignment/register.py": "pairing role: moving platform",
        "src/merxen/analysis/clustering_squidpy.py": (
            "MERSCOPE-moving aligned layers (pairing role)"
        ),
        "src/merxen/config.py": "pairing roles of AlignmentConfig",
        "src/merxen/distance_from_object/pipeline.py": (
            "MERSCOPE-moving aligned layers (pairing role)"
        ),
        "src/merxen/enrichment/enrich.py": (
            "dispatch that raises for an unknown platform, the raw Xenium reader, "
            "and the registry family checks"
        ),
        "src/merxen/io/builders/pipeline.py": (
            "builder dispatch that raises for an unknown platform"
        ),
        "src/merxen/segmentation/pipeline.py": (
            "dispatch that raises for an unknown platform; Literals guard the rest"
        ),
        "src/merxen/visualization/density_overview.py": "display only",
        "src/merxen/visualization/sanity_plots.py": (
            "display dispatch that raises for an unknown dataset, aligned-layer display"
        ),
        "workflows/main.nf": (
            "builders that raise for an unknown platform, MERSCOPE- or Xenium-only "
            "features, paired roles and the analysis_mode map (kept in step by "
            "test_analysis_mode_parity)"
        ),
        "scripts/acceptance/segmentation_comparison.py": (
            "RCA acceptance script for paired MERSCOPE/XENIUM sections (Xenium QV "
            "filter and panel)"
        ),
        "scripts/acceptance/shadow_baselines.py": (
            "RCA acceptance script: Xenium fixed frame of a pair (pairing role)"
        ),
        "src/merxen/annotation/panel.py": (
            "map_first panel modes of the MERSCOPE/XENIUM targeted panels (set c "
            "needs both pseudobulks, Xenium symbols first); ATERA rows are kept "
            "off map_first until AT-11 (RCA on Atera; main.nf ATERA gate)"
        ),
        "src/merxen/annotation/pipeline.py": (
            "map_first pairing roles (Xenium fixed frame, MERSCOPE moving) and the "
            "MERSCOPE+XENIUM cross-platform pair; ATERA rows are kept off "
            "map_first until AT-11 (RCA on Atera; main.nf ATERA gate)"
        ),
        "src/merxen/annotation/report_scoring.py": (
            "fixed scoring datasets of the human acceptance pairs (aliases)"
        ),
        "src/merxen/annotation/sim_inputs.py": (
            "panel chemistry detection (MERSCOPE, else Xenium Prime gene lists); "
            "ATERA rows are kept off map_first until AT-11 (RCA on Atera; main.nf "
            "ATERA gate)"
        ),
    },
    "native_image_names": {
        "scripts/public_data/xenium_hippocampus/make_samplesheet_dg_lifespan.py": (
            "public-data intake shim: writes and checks the vendor Xenium"
            " output layout (file names), not registry layers"
        ),
        "scripts/public_data/xenium_hippocampus/prepare_dg_lifespan.py": (
            "public-data intake shim: writes and checks the vendor Xenium"
            " output layout (file names), not registry layers"
        ),
        "scripts/public_data/xenium_hippocampus/prepare_ki_hippocampus.py": (
            "public-data intake shim: writes and checks the vendor Xenium"
            " output layout (file names), not registry layers"
        ),
        "scripts/manual_registration/export_reference_image.py": (
            "standalone export script; --image-key is a user-overridable default"
        ),
        "src/merxen/alignment/frames.py": (
            "image preference list with a Xenium MIP fallback (pairing roles)"
        ),
        "src/merxen/config.py": "alignment and registration defaults, reader option",
        "src/merxen/enrichment/enrich.py": "raw Xenium reader option",
        "src/merxen/image_registration.py": "docstring",
        "src/merxen/io/builders/xenium.py": (
            "Xenium reader option; pre-2.0 single-channel images named DAPI"
        ),
        "src/merxen/io/image_source.py": "the MERSCOPE image name constant",
        "src/merxen/segmentation/pipeline.py": (
            "dispatch that raises for an unknown platform, error messages, and the "
            "Xenium nuclei arm"
        ),
        "src/merxen/visualization/sanity_plots.py": "display image selection",
    },
}

# Category -> file -> sorted stripped matching lines (files only for file-level
# categories). Generated from the tree; edit by hand with a reason above.
PINNED: dict[str, dict[str, tuple[str, ...]]] = {
    "shape_names": {
        "src/merxen/alignment/legacy_features.py": (
            '"merscope_cell_boundaries",',
            '"merscope_cell_boundaries_aligned_nonrigid",',
            '"xenium_cell_boundaries",',
            '"xenium_cell_boundaries_aligned_nonrigid",',
        ),
        "src/merxen/analysis/clustering_squidpy.py": (
            '"merscope_cell_boundaries",',
            '"merscope_cell_boundaries_aligned_nonrigid",',
            '"xenium_cell_boundaries",',
            '"xenium_nucleus",',
            '"xenium_nucleus_aligned_nonrigid",',
        ),
        "src/merxen/analysis/pciseq.py": (
            '"original_seg": ("xenium_cell_boundaries", "merscope_cell_boundaries"),',
        ),
        "src/merxen/visualization/sanity_plots.py": (
            '"merscope_cell_boundaries",',
            '"merscope_cell_boundaries",',
            '"merscope_cell_boundaries": ("Original segmentation", "#ff7f0e"),',
            '"xenium_cell_boundaries",',
            '"xenium_cell_boundaries",',
            '"xenium_cell_boundaries": ("Original segmentation", "#ff7f0e"),',
        ),
    },
    "built_shape_names": {
        "src/merxen/masks.py": ('f"{ORIGINAL_ID_NAMESPACE}_nucleus",',),
    },
    "vendor_nucleus_names": {
        "scripts/public_data/xenium_hippocampus/fetch_dg_lifespan.py": (
            '"nucleus_boundaries.parquet.gz": "nucleus_boundaries",',
        ),
        "scripts/public_data/xenium_hippocampus/prepare_dg_lifespan.py": (
            '"nucleus_boundaries": "nucleus_boundaries.parquet",',
            '"nucleus_boundaries.parquet": ("cell_id", "vertex_x", "vertex_y"),',
            "``nucleus_boundaries.parquet``, ``transcripts.parquet``: decompressed from",  # noqa: E501
            'nucleus_boundaries, index, image.pixel_size_um, label_by="polygon"',
            "nucleus_boundaries: Nucleus polygon parquet.",
            "nucleus_boundaries: Path,",
            'nucleus_boundaries=out_dir / "nucleus_boundaries.parquet",',
            "path: ``cell_boundaries.parquet`` or ``nucleus_boundaries.parquet``.",
        ),
        "scripts/public_data/xenium_hippocampus/prepare_ki_hippocampus.py": (
            '"nucleus_boundaries.parquet",',
            '"nucleus_boundaries.parquet": ("cell_id", "vertex_x", "vertex_y"),',
            'for name in ("cell_boundaries.parquet", "nucleus_boundaries.parquet"):',
        ),
        "src/merxen/analysis/clustering_squidpy.py": (
            '"nucleus_boundaries",',
            '"xenium_nucleus",',
        ),
        "src/merxen/config.py": (
            "nucleus_boundaries: bool = True",
            "nucleus_boundaries: bool = True",
        ),
        "src/merxen/enrichment/enrich.py": (
            '["nucleus_boundaries", XENIUM_OLD_NUCLEUS_SHAPE_NAME],',
            'csv_path = xenium_dir / "nucleus_boundaries.csv.gz"',
            "nucleus_boundaries=False,",
        ),
        "src/merxen/io/atera_reader.py": (
            '("nucleus_boundaries", "masks/0", cells.n_nuclei),',
            '("nucleus_boundaries", cells.nucleus_vertices, '
            "cells.nucleus_num_vertices),",
            '``{"cell_boundaries": ..., "nucleus_boundaries": ...}``. Both carry',
        ),
        "src/merxen/io/builders/atera.py": (
            'NUCLEUS_SHAPES_KEY: Final = "nucleus_boundaries"',
            "NUCLEUS_SHAPES_KEY: build_config.nucleus_boundaries,",
        ),
        "src/merxen/io/builders/xenium.py": (
            "nucleus_boundaries=build_config.nucleus_boundaries,",
        ),
        "src/merxen/masks.py": ('("nucleus_boundaries",),',),
        "src/merxen/segmentation/native.py": (
            '("shapes", "nucleus_boundaries"): ("shapes", NATIVE_NUCLEUS_SHAPES_KEY),',
            "``cell_boundaries`` -> ``atera_cell_boundaries``, ``nucleus_boundaries`` ->",  # noqa: E501
        ),
        "src/merxen/segmentation/pipeline.py": ("nucleus_boundaries=False,",),
    },
    "nucleus_joins": {
        "scripts/public_data/xenium_hippocampus/prepare_dg_lifespan.py": (
            '"nucleus_area",',
            'if "nucleus_area" in cells:',
            'summary[:, 5] = cells["nucleus_area"].to_numpy()',
        ),
        "scripts/acceptance/atera_build.py": (
            'VENDOR_OBS_NOT_COPIED = frozenset({"cell_area", "nucleus_area"})',
            '"nucleus_area": "nucleus_area",',
        ),
        "scripts/acceptance/make_atera_crop.py": (
            'nucleus_area = cells.summary["nucleus_area"].to_numpy(np.float64)',
            "nucleus_ok = ~has_nucleus | (largest == np.rint(nucleus_area / "
            "pixel_area))",
        ),
        "src/merxen/analysis/clustering_squidpy.py": (
            '"nucleus_area",',
            '"nucleus_area": grouped.max(),',
            '"nucleus_area_total": grouped.sum(min_count=1),',
            'NUCLEUS_METRIC_COLUMNS = ("nucleus_area", "nucleus_area_total", '
            '"nucleus_count")',
            "``instance_id``: ``nucleus_area`` is the largest nucleus of the cell,",
            "``nucleus_area_total`` the summed area and ``nucleus_count`` the "
            "number of",
            '_apply_area_metric(adata, nucleus_metrics, column="nucleus_area")',
            '_obs_numeric(adata, "nucleus_area"),',
            'adata.obs["nucleus_area"] = np.nan',
            "def _choose_nucleus_shape_key(",
            'if "nucleus_area" not in adata.obs:',
            "nucleus_shape_key, nucleus_join = _choose_nucleus_shape_key(",
        ),
        "src/merxen/io/atera_reader.py": (
            "``cell_summary.nucleus_area`` describes only the largest nucleus of a",
            "def atera_nucleus_area_total(",
            "if nucleus_area_total is None:",
            "nucleus_area_total = atera_nucleus_area_total(bundle, cells)",
            "nucleus_area_total: Total nucleus area per cell; computed from",
            "nucleus_area_total: np.ndarray | None = None,",
            'obs["nucleus_area"] = summary["nucleus_area"].to_numpy(np.float64)',
            'obs["nucleus_area_total"] = '
            "np.asarray(nucleus_area_total, dtype=np.float64)",
            "def nucleus_area_total_from_counts(",
            "exported (for :func:`nucleus_area_total_from_counts`).",
            "return nucleus_area_total_from_counts(counter.counts, cells, "
            "bundle.pixel_size_um)",
            '{"cell_centroid_x", "cell_centroid_y", "cell_area", "nucleus_area", '
            '"nucleus_count"}',
        ),
        "src/merxen/io/builders/atera.py": (
            "ar.nucleus_area_total_from_counts(",
            "nucleus_area_total=(",
        ),
        "src/merxen/io/spatialdata_io.py": (
            'if "cell" in str(key).lower() and "nucleus" not in str(key).lower()',
        ),
        "src/merxen/visualization/sanity_plots.py": (
            'if "nucleus" not in str(key).lower() and str(key) != "cell_boundaries":',
            'if str(key).endswith(suffix) and "nucleus" not in str(key).lower():',
        ),
    },
    "nuclei_mask_consumers": {
        "src/merxen/cli/__init__.py": (),
        "src/merxen/segmentation/hitl/seed_inputs.py": (),
        "workflows/lib/CellposeHitlSettings.groovy": (),
        "src/merxen/cli/run_segmentation.py": (),
        "src/merxen/config.py": (),
        "src/merxen/enrichment/enrich.py": (),
        "src/merxen/masks.py": (),
        "src/merxen/registration_quality.py": (),
        "src/merxen/segmentation/pipeline.py": (),
        "src/merxen/viewer_cache/build.py": (),
        "workflows/main.nf": (),
        "workflows/modules/enrichment.nf": (),
        "workflows/modules/segmentation.nf": (),
        "scripts/acceptance/segmentation_comparison.py": (),
    },
    "platform_equality": {
        "src/merxen/segmentation/hitl/import_annotations.py": (
            'platform not in {"ATERA", "XENIUM"}',
        ),
        "workflows/lib/CellposeHitlSettings.groovy": (
            "if (!(spec.platform in ['ATERA', 'XENIUM']) || "
            "!(spec.species in ['human', 'mouse'])) {",
        ),
        "src/merxen/segmentation/hitl/regions.py": (
            'if self.schema_version != 1 or self.platform not in ("ATERA", "XENIUM"):',
        ),
        "scripts/cleanup_assignment_tables_and_outputs.py": (
            'if mode == "xenium":',
            'if mode in {"merscope", "merfish"}:',
        ),
        "src/merxen/alignment/dapi.py": (
            'if str(platform or "").strip().upper() != "MERSCOPE":',
        ),
        "src/merxen/alignment/frames.py": ('if platform == "MERSCOPE"',),
        "src/merxen/alignment/manifest.py": (
            'if str(result.metadata.get("moving_platform", "MERSCOPE")) == "MERSCOPE"',
        ),
        "src/merxen/alignment/materialize.py": (
            'if cfg.fixed_platform != "XENIUM" or cfg.moving_platform != "MERSCOPE":',
        ),
        "src/merxen/alignment/pipeline.py": (
            'if config.fixed_platform == "XENIUM"',
            'if config.moving_platform == "MERSCOPE"',
            'if moving_platform == "MERSCOPE"',
        ),
        "src/merxen/alignment/register.py": ('if cfg.moving_platform == "MERSCOPE":',),
        "src/merxen/analysis/clustering_squidpy.py": (
            'if platform.upper() == "MERSCOPE" and aligned_preferred in sdata_obj.shapes:',  # noqa: E501
            'if platform.upper() == "MERSCOPE" and aligned_region in sdata_obj.shapes:',
        ),
        "src/merxen/config.py": (
            'self.fixed_platform != "XENIUM" or self.moving_platform != "MERSCOPE"',
        ),
        "src/merxen/distance_from_object/pipeline.py": (
            'if platform.upper() == "MERSCOPE" and aligned in sdata_obj.shapes:',
        ),
        "src/merxen/enrichment/enrich.py": (
            'elif platform == "XENIUM":',
            'if config.platform.upper() == "XENIUM":',
            'if platform == "MERSCOPE":',
            'if platform == "MERSCOPE":',
            'if platform == "XENIUM":',
            'if spec.family == "MERSCOPE":',
            'if spec.family == "MERSCOPE":',
            'if spec.family == "MERSCOPE":',
        ),
        "src/merxen/io/builders/pipeline.py": (
            'elif platform == "XENIUM":',
            'if cached is None or config.platform.upper() != "MERSCOPE":',
            'if platform == "MERSCOPE":',
            'platform.upper() == "MERSCOPE"',
        ),
        "src/merxen/segmentation/pipeline.py": (
            'elif platform == "MERSCOPE":',
            'elif platform == "XENIUM":',
            'if is_experimental_arm(config) and dataset.platform == "XENIUM":',
            'if platform == "MERSCOPE":',
            'if platform == "MERSCOPE":',
            'if platform == "XENIUM":',
            'if platform == "XENIUM":',
        ),
        "src/merxen/visualization/density_overview.py": (
            'if dataset_label == "MERSCOPE"',
        ),
        "src/merxen/visualization/sanity_plots.py": (
            'if dataset == "MERSCOPE":',
            'if dataset == "XENIUM":',
            'if dataset_name.upper() == "MERSCOPE" and prefer_aligned_vectors:',
            'if dataset_name.upper() == "MERSCOPE":',
        ),
        "workflows/main.nf": (
            '!(settings.run_register_images && platform == "XENIUM")',
            '.filter { _key, _pairId, platform, _zarrPath -> platform == "MERSCOPE" }',
            '.filter { _key, _pairId, platform, _zarrPath -> platform == "XENIUM" }',
            'if (analysisMode == "merscope") {',
            'if (analysisMode == "xenium") {',
            'if (platform == "MERSCOPE") {',
            'if (platform == "MERSCOPE") {',
            'if (platform == "XENIUM") {',
            'if (platform == "XENIUM") {',
            'settings.paired_mode && platform == "MERSCOPE"',
            'settings.paired_mode && platform == "XENIUM"',
        ),
        "scripts/acceptance/segmentation_comparison.py": (
            'if platform == "MERSCOPE"',
            'min_qv = args.xenium_min_qv if platform == "XENIUM" else None',
            'min_qv = args.xenium_min_qv if platform == "XENIUM" else None',
            'panel_path = xenium_panels.get(pair) if platform == "XENIUM" else None',
            'part["xenium"] = part["platform"] == "XENIUM"',
        ),
        "scripts/acceptance/shadow_baselines.py": (
            'aligned_frame=platform == "XENIUM" or "_aligned" in shape_key,',
            'if merscope.aligned_frame and mask.fixed_platform == "XENIUM":',
        ),
        "src/merxen/annotation/panel.py": (
            'for platform in ("MERSCOPE", "XENIUM"):',
            'order = sorted(symbol_maps, key=lambda platform: platform != "XENIUM")',
        ),
        "src/merxen/annotation/pipeline.py": (
            '(item.h5ad_path for item in inputs if item.platform == "MERSCOPE"),',
            'for name in ("MERSCOPE", "XENIUM")',
            'if platform not in {"MERSCOPE", "XENIUM"}:',
            (
                'return str(platform).upper() == "XENIUM" or "_aligned" in '
                'str(shape_key or "")'
            ),
        ),
        "src/merxen/annotation/report_scoring.py": (
            'for platform in ("MERSCOPE", "XENIUM")',
        ),
        "src/merxen/annotation/sim_inputs.py": (
            'and platform_name in ("", "XENIUM")',
            'if platform_name == "MERSCOPE":',
        ),
    },
    "native_image_names": {
        "scripts/public_data/xenium_hippocampus/make_samplesheet_dg_lifespan.py": (
            "#: ``morphology_focus/`` folder ``DAPI`` whatever its OME-XML says, so this is",  # noqa: E501
        ),
        "scripts/public_data/xenium_hippocampus/prepare_dg_lifespan.py": (
            '"morphology_focus_0000.ome.tif is a MerXen projection of the deposited "',
            "- ``morphology_focus/morphology_focus_0000.ome.tif``: a 2-D projection of the",  # noqa: E501
            'FOCUS_FILE: Final = Path("morphology_focus") / "morphology_focus_0000.ome.tif"',  # noqa: E501
            "spatialdata-io names a one-file XOA 2/3 ``morphology_focus/``",
        ),
        "scripts/public_data/xenium_hippocampus/prepare_ki_hippocampus.py": (
            '"morphology_focus.ome.tif",',
            "# spatialdata-io gives a pre-2.0 morphology_focus no channel names.",
            "#: fixed build options (labels, boundaries, transcripts, morphology_focus).",  # noqa: E501
            "The ``morphology_focus.ome.tif`` of XOA below 2.0 holds one channel (DAPI) that",  # noqa: E501
            'f"{section}: morphology_focus.ome.tif has {size_c} channels; one "',
            "morphology_channels: OME channel names of ``morphology_focus.ome.tif``.",
            'size_c, names, physical = _ome_channels(folder / "morphology_focus.ome.tif")',  # noqa: E501
        ),
        "scripts/manual_registration/export_reference_image.py": (
            "``morphology_focus`` image lives inside a SpatialData store, so a ground-truth",  # noqa: E501
            "Xenium stores keep ``morphology_focus`` in pixel units (an identity ``global``",  # noqa: E501
            'parser.add_argument("--image-key", default="morphology_focus")',
            '"is the identity, as for Xenium morphology_focus: 0.2125).",',
        ),
        "src/merxen/alignment/frames.py": (
            'else ["morphology_focus", "morphology_mip"]',
        ),
        "src/merxen/config.py": (
            'fixed_image_key: str = "morphology_focus"',
            "morphology_focus: bool = True",
            'output_image_key: str = "MERSCOPE_z_projection_aligned_nonrigid"',
            "reference image (Xenium ``morphology_focus``).",
            'reference_image_key: str = "morphology_focus"',
            'source_image_key: str = "MERSCOPE_z_projection"',
        ),
        "src/merxen/enrichment/enrich.py": ("morphology_focus=True,",),
        "src/merxen/image_registration.py": (
            "image level-0 pixels to reference (``morphology_focus``) level-0 pixels. The",  # noqa: E501
        ),
        "src/merxen/io/builders/xenium.py": (
            'PRE_V2_MORPHOLOGY_IMAGES: Final = ("morphology_focus", "morphology_mip")',
            "``morphology_focus.ome.tif`` (and ``morphology_mip.ome.tif``) without",
            "morphology_focus=build_config.morphology_focus,",
        ),
        "src/merxen/io/image_source.py": (
            'MERSCOPE_ZPROJ_IMAGE_NAME = "MERSCOPE_z_projection"',
        ),
        "src/merxen/segmentation/pipeline.py": (
            '"Custom MERSCOPE models require native MERSCOPE_z_projection"',
            '"morphology_focus"',
            "morphology_focus=True,",
            'nuclei_config.dataset.image_key = "morphology_focus"',
            'raise ValueError("Custom Xenium models require native morphology_focus")',
        ),
        "src/merxen/visualization/sanity_plots.py": (
            '"morphology_focus"',
            'if "MERSCOPE_z_projection" in images:',
            'if "morphology_focus" in sdata_obj.images',
            'return "MERSCOPE_z_projection"',
        ),
    },
}


def _scanned_files() -> list[Path]:
    """Return every scanned source file except the registries, sorted."""
    files: set[Path] = set()
    for pattern in _SCANNED_GLOBS:
        files.update(path for path in _REPO_ROOT.glob(pattern) if path.is_file())
    return sorted(
        path
        for path in files
        if path.relative_to(_REPO_ROOT).as_posix() not in _REGISTRY_FILES
    )


def scan(category: SiteCategory) -> dict[str, tuple[str, ...]]:
    """Return ``relative path -> sorted stripped matching lines`` of a category.

    Args:
        category: The pattern to look for.

    Returns:
        The matching files; for a file-level category each maps to ``()``.
    """
    regex = re.compile(category.pattern)
    found: dict[str, tuple[str, ...]] = {}
    for path in _scanned_files():
        lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if regex.search(line)
        ]
        if lines:
            relative = path.relative_to(_REPO_ROOT).as_posix()
            found[relative] = () if category.file_level else tuple(sorted(lines))
    return found


def test_scan_covers_the_workflow_and_package() -> None:
    """The scan sees the package, the workflow and its library, not the registry."""
    scanned = {path.relative_to(_REPO_ROOT).as_posix() for path in _scanned_files()}
    for expected in (
        "src/merxen/enrichment/enrich.py",
        "workflows/main.nf",
        "workflows/modules/qc.nf",
        "workflows/lib/CellposeSettings.groovy",
        "workflows/acceptance/hitl0_tile_check.nf",
    ):
        assert expected in scanned
    assert not scanned & _REGISTRY_FILES


@pytest.mark.parametrize("category", CATEGORIES, ids=lambda item: item.name)
def test_platform_sites_match_the_pin(category: SiteCategory) -> None:
    """Every site that names a platform value is pinned with a reason."""
    found = scan(category)
    pinned = PINNED[category.name]
    problems: list[str] = []
    for path in sorted(set(found) | set(pinned)):
        added = Counter(found.get(path, ())) - Counter(pinned.get(path, ()))
        removed = Counter(pinned.get(path, ())) - Counter(found.get(path, ()))
        if path not in pinned:
            problems.append(f"new file {path}")
        elif path not in found:
            problems.append(f"no longer matches: {path}")
        problems.extend(f"new line in {path}: {line}" for line in added.elements())
        problems.extend(f"line gone from {path}: {line}" for line in removed.elements())
    assert not problems, (
        f"{category.name} sites changed:\n  "
        + "\n  ".join(problems)
        + "\nLook the value up in the platform registry (src/merxen/platforms.py, "
        "workflows/lib/PlatformSpec.groovy) instead, or update PINNED and REASONS "
        "in tests/test_platform_sites.py with the reason the site must stay."
    )


@pytest.mark.parametrize("category", CATEGORIES, ids=lambda item: item.name)
def test_every_pinned_file_has_a_reason(category: SiteCategory) -> None:
    """A pinned site without a reason is an unreviewed site."""
    reasons = REASONS.get(category.name, {})
    missing = sorted(set(PINNED[category.name]) - set(reasons))
    stale = sorted(set(reasons) - set(PINNED[category.name]))
    assert not missing, f"{category.name}: no reason for {missing}"
    assert not stale, f"{category.name}: reasons for unpinned files {stale}"
