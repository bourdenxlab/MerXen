"""Pre-build the derived caches the napari comparison viewer builds on the fly.

The viewer ``napari-compare-xenium-merscope`` lazily rasterizes segmentation
masks and materializes image/label pyramids the first time a dataset is opened.
That work is slow and identical for every viewing session, so this package
reproduces it during preprocessing and writes the results into the enriched
``latest_spatialdata.zarr`` store using the *exact* element keys and completion
markers the viewer checks -- so the viewer trusts the pre-built caches and skips
the on-the-fly build entirely.

The format the viewer expects is duplicated here (see :mod:`merxen.viewer_cache.format`)
rather than imported, because the pipeline must not depend on napari/Qt. The
pinned :data:`~merxen.viewer_cache.format.VIEWER_DERIVED_CACHE_VERSION` and the
golden-value test in ``tests/test_viewer_cache`` guard against silent drift from
the viewer's format.
"""

from __future__ import annotations

from merxen.viewer_cache.build import build_viewer_caches

__all__ = ["build_viewer_caches"]
