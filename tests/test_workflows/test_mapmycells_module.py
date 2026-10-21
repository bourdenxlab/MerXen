"""Workflow text smoke tests for the MapMyCells Nextflow module."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = REPO_ROOT / "workflows"
DWIGHT_SITE_CONFIG_EXAMPLE = WORKFLOWS_DIR / "conf" / "dwight.site.config.example"


def test_gene_id_fallback_param_reaches_the_mapmycells_config() -> None:
    """The fallback table param should be defined, validated and passed on."""
    base_config = (REPO_ROOT / "workflows" / "nextflow.config").read_text()
    module_text = (REPO_ROOT / "workflows" / "modules" / "mapmycells.nf").read_text()
    main_text = (REPO_ROOT / "workflows" / "main.nf").read_text()

    assert re.search(r"^\s*annotation_gene_id_fallback_csv = null$", base_config, re.M)
    assert '"gene_id_fallback_csv": ${geneIdFallbackJson}' in module_text
    assert "params.annotation_gene_id_fallback_csv" in module_text
    assert "params.annotation_gene_id_fallback_csv" in main_text
    assert "MAPMYCELLS gene-ID fallback table" in main_text


def test_dwight_gene_id_fallback_points_at_a_local_whb_table() -> None:
    """Dwight should use a local WHB gene table on SSD1, never a download URL.

    The value is a host location, so the Dwight site config sets it; the
    template (placeholders mirroring the real layout) is what is tracked.
    """
    dwight = (WORKFLOWS_DIR / "conf" / "dwight.config").read_text()
    assert "annotation_gene_id_fallback_csv" not in dwight
    match = re.search(
        r'^\s*annotation_gene_id_fallback_csv = "([^"]+)"$',
        DWIGHT_SITE_CONFIG_EXAMPLE.read_text(),
        re.M,
    )
    assert match is not None
    path = match.group(1)
    assert path.startswith("/path/to/legacy-disk/MerXen/mapmycells/abc_whb/")
    assert "WHB-10Xv3" in path
    assert "://" not in path
