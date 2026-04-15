"""Shared type aliases for MerXen."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

# A 2D labeled mask array (height x width) with integer cell IDs
LabelMask: TypeAlias = npt.NDArray[np.uint32]

# Affine transform components: (scale_x, shear_x, offset_x)
AffineComponent: TypeAlias = tuple[float, float, float]

# Image source dict returned by build_image_source
ImageSource: TypeAlias = dict
