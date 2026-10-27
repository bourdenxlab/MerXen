"""Store-attrs contract of manual landmark registration.

Constants, pydantic payload models, canonical JSON and digest helpers and the
root-attrs reader/writer shared by MerXen and the napari viewer. The values of
the constants and the digest recipes are pinned by the golden vectors
(``tests/test_data/manual_registration/golden_v1.json``).

This module imports nothing from the rest of the package. ``zarr`` and
``merxen.io.spatialdata_io`` are imported inside the functions that need them, so
the numpy-only geometry modules stay light.
"""

from __future__ import annotations

import colorsys
import hashlib
import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION: Final = 1
STORE_ATTR: Final = "merxen_manual_registration_store"
LABELLING_ATTR: Final = "merxen_manual_registration"
RESULT_ATTR: Final = "merxen_manual_registration_result"
STORE_SCHEMA: Final = "merxen.manual_registration.store"
LABELLING_SCHEMA: Final = "merxen.manual_registration.labelling"
RESULT_SCHEMA: Final = "merxen.manual_registration.result"
# Used by the viewer; recorded here so the golden file pins it.
LAYER_METADATA_KEY: Final = "merxen_manual_registration"
MIN_LANDMARK_PAIRS: Final = 8
DEFAULT_OUTLIER_FLOOR_UM: Final = 10.0
OUTLIER_K: Final = 3.0
MAX_OUTLIER_FRACTION: Final = 0.25
MIN_USED_PAIRS: Final = 6
CROP_SNAP_PX: Final = 64
CROP_MIN_SIZE_PX: Final = 1024
DATASET_ID_PATTERN: Final = r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"
DIGEST_PATTERN: Final = r"^sha256:[0-9a-f]{64}$"
COLOUR_PATTERN: Final = r"^#[0-9a-f]{6}$"
CONVENTIONS: Final[dict[str, str]] = {
    "pixel": "level0_xy_integer_is_pixel_centre",
    "matrix": "moving_level0_px_xy_to_reference_level0_px_xy",
    "rotation": "positive_ccw_on_screen_y_down",
    "crop": "half_open_level0_px",
}

_GOLDEN_RATIO_CONJUGATE: Final = 0.6180339887498949
_T = TypeVar("_T")


class UnsupportedSchemaVersionError(ValueError):
    """A payload was written by a newer MerXen or viewer."""


def check_schema_version(payload: Mapping[str, Any], *, attr: str) -> None:
    """Check the ``schema_version`` of a root-attrs payload.

    Args:
        payload: The raw attrs mapping.
        attr: The attrs key the payload was read from (used in messages).

    Raises:
        ValueError: If the version is not a positive integer.
        UnsupportedSchemaVersionError: If it is newer than ``SCHEMA_VERSION``.
    """
    version = payload.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError(
            f"{attr}: schema_version must be a positive integer, got {version!r}"
        )
    if version > SCHEMA_VERSION:
        raise UnsupportedSchemaVersionError(
            f"{attr}: schema_version {version} was written by a newer "
            "MerXen/viewer; update"
        )


def wrap_angle_deg(angle_deg: float) -> float:
    """Wrap an angle to the half-open interval (-180, 180] degrees.

    Raises:
        ValueError: If the angle is not finite.
    """
    if not math.isfinite(angle_deg):
        raise ValueError("angle must be finite")
    wrapped = math.fmod(angle_deg, 360.0)
    if wrapped <= -180.0:
        wrapped += 360.0
    elif wrapped > 180.0:
        wrapped -= 360.0
    return wrapped


def _all_finite(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, int | float):
        return math.isfinite(value)
    if isinstance(value, list | tuple):
        return all(_all_finite(item) for item in value)
    return True


def _finite(value: _T) -> _T:
    """Field validator body: reject NaN and infinities in floats and sequences."""
    if not _all_finite(value):
        raise ValueError("must be finite")
    return value


class CropRect(BaseModel):
    """Half-open crop ``arr[y0:y1, x0:x1]`` in level-0 pixels of the full image."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    x0: int = Field(ge=0)
    y0: int = Field(ge=0)
    x1: int
    y1: int

    @model_validator(mode="after")
    def _validate_order(self: CropRect) -> CropRect:
        if not (self.x0 < self.x1 and self.y0 < self.y1):
            raise ValueError(
                f"crop must satisfy x0 < x1 and y0 < y1, got {self.as_tuple()}"
            )
        return self

    def width(self) -> int:
        """Return the crop width in pixels."""
        return self.x1 - self.x0

    def height(self) -> int:
        """Return the crop height in pixels."""
        return self.y1 - self.y0

    def as_tuple(self) -> tuple[int, int, int, int]:
        """Return ``(x0, y0, x1, y1)``."""
        return (self.x0, self.y0, self.x1, self.y1)

    def validate_within(self, shape_yx: tuple[int, int]) -> None:
        """Check that the crop lies inside an image of shape ``(H, W)``.

        Raises:
            ValueError: If it exceeds the image.
        """
        height, width = shape_yx
        if self.x1 > width or self.y1 > height:
            raise ValueError(
                f"crop {self.as_tuple()} exceeds the image of shape (H, W) = "
                f"{tuple(shape_yx)}"
            )


class CoarseTransform(BaseModel):
    """Coarse moving-to-reference transform state (rotation, mirror, offset)."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    rotation_deg: float = 0.0
    mirror_x: bool = False
    offset_um: tuple[float, float] = (0.0, 0.0)

    @field_validator("rotation_deg")
    @classmethod
    def _wrap_rotation(cls: type[CoarseTransform], value: float) -> float:
        return wrap_angle_deg(value)

    @field_validator("offset_um")
    @classmethod
    def _finite_offset(
        cls: type[CoarseTransform], value: tuple[float, float]
    ) -> tuple[float, float]:
        return _finite(value)


class LandmarkPair(BaseModel):
    """One reference/moving landmark pair, in level-0 pixels of the full images."""

    model_config = ConfigDict(extra="ignore")

    id: int = Field(ge=1)
    colour: str = Field(pattern=COLOUR_PATTERN)
    enabled: bool = True
    reference_xy_px: tuple[float, float]
    moving_xy_px: tuple[float, float]

    @field_validator("reference_xy_px", "moving_xy_px")
    @classmethod
    def _finite_xy(
        cls: type[LandmarkPair], value: tuple[float, float]
    ) -> tuple[float, float]:
        return _finite(value)


class LabellingImage(BaseModel):
    """Shape and pixel size of one full image as seen by the labelling tool."""

    model_config = ConfigDict(extra="ignore")

    shape_yx: tuple[int, int]
    pixel_size_um: float = Field(gt=0.0)

    @field_validator("shape_yx")
    @classmethod
    def _positive_shape(
        cls: type[LabellingImage], value: tuple[int, int]
    ) -> tuple[int, int]:
        if value[0] <= 0 or value[1] <= 0:
            raise ValueError(f"shape_yx must be positive, got {value}")
        return value

    @field_validator("pixel_size_um")
    @classmethod
    def _finite_pixel_size(cls: type[LabellingImage], value: float) -> float:
        return _finite(value)


class LabellingImages(BaseModel):
    """The two images of a labelling session."""

    model_config = ConfigDict(extra="ignore")

    reference: LabellingImage
    moving: LabellingImage


class LabellingCoarse(CoarseTransform):
    """Coarse state as saved by the viewer, with an informative matrix."""

    matrix_xy: list[list[float]] | None = None  # informative only

    @field_validator("matrix_xy")
    @classmethod
    def _finite_matrix(
        cls: type[LabellingCoarse], value: list[list[float]] | None
    ) -> list[list[float]] | None:
        return _finite(value)


class LabellingCrops(BaseModel):
    """Optional crops of the two images (``None`` = no crop)."""

    model_config = ConfigDict(extra="ignore")

    reference: CropRect | None = None
    moving: CropRect | None = None


class LabellingLandmarks(BaseModel):
    """The landmark pairs and the counters needed to continue labelling."""

    model_config = ConfigDict(extra="ignore")

    next_id: int = Field(ge=1)
    colour_seed: int = Field(ge=0)
    pairs: list[LandmarkPair] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_ids(self: LabellingLandmarks) -> LabellingLandmarks:
        ids = [pair.id for pair in self.pairs]
        duplicates = {pair_id for pair_id in ids if ids.count(pair_id) > 1}
        if duplicates:
            raise ValueError(f"duplicate landmark ids: {sorted(duplicates)}")
        if ids and self.next_id <= max(ids):
            raise ValueError("next_id must exceed every pair id")
        return self


class WrittenBy(BaseModel):
    """Software that wrote a payload."""

    model_config = ConfigDict(extra="ignore")

    software: str
    version: str


class LabellingAttrs(BaseModel):
    """Root attrs ``LABELLING_ATTR``: the viewer's labelling session (single writer)."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    # ``schema`` shadows BaseModel.schema, hence the alias.
    schema_: Literal["merxen.manual_registration.labelling"] = Field(alias="schema")
    schema_version: int
    status: Literal["draft", "finished"]
    revision: int = Field(ge=1)
    updated_at: str
    finished_at: str | None = None
    written_by: WrittenBy
    store_grid_digest: str | None = Field(default=None, pattern=DIGEST_PATTERN)
    images: LabellingImages
    coarse: LabellingCoarse
    crops: LabellingCrops = Field(default_factory=LabellingCrops)
    landmarks: LabellingLandmarks
    content_sha256: str | None = Field(default=None, pattern=DIGEST_PATTERN)
    # Informative: the viewer's floor at save time; not part of the content digest.
    outlier_floor_um: float | None = Field(default=None, gt=0)

    @field_validator("schema_version")
    @classmethod
    def _check_version(cls: type[LabellingAttrs], value: int) -> int:
        check_schema_version({"schema_version": value}, attr=LABELLING_ATTR)
        return value

    @field_validator("outlier_floor_um")
    @classmethod
    def _finite_floor(cls: type[LabellingAttrs], value: float | None) -> float | None:
        return _finite(value)

    def enabled_pairs(self) -> list[LandmarkPair]:
        """Return the enabled landmark pairs, in stored order."""
        return [pair for pair in self.landmarks.pairs if pair.enabled]


class OmeTiffRecord(BaseModel):
    """The OME-TIFF a prepared image was read from or written to."""

    model_config = ConfigDict(extra="ignore")

    path: str
    fingerprint: str = Field(pattern=DIGEST_PATTERN)
    written_by_prepare: bool
    level_shapes_yx: list[tuple[int, int]]
    exact_level_factors: bool


class SourceRecord(BaseModel):
    """The user-supplied source file of a prepared image."""

    model_config = ConfigDict(extra="ignore")

    format: Literal["czi", "ome_tiff"]
    path: str
    fingerprint: str = Field(pattern=DIGEST_PATTERN)
    size_bytes: int = Field(ge=0)


class CziRecord(BaseModel):
    """CZI-specific provenance of a prepared image."""

    model_config = ConfigDict(extra="ignore")

    scene_index: int = Field(ge=0)
    scene_name: str | None = None
    bbox_xywh: tuple[int, int, int, int]
    z_planes_projected: int = Field(ge=1)
    # None when the CZI metadata lacks them.
    pixel_type: str | None = None
    compression: str | None = None


class StoreImageRecord(BaseModel):
    """One image of a manual-registration store (prepared or result variant)."""

    model_config = ConfigDict(extra="ignore")

    element: str
    role: Literal["reference", "moving"]
    shape_yx: tuple[int, int]
    dtype: str
    pixel_size_um: float = Field(gt=0.0)
    channel_names: list[str]
    registration_channel: str
    # prepared-store fields
    pixel_size_source: Literal["metadata", "override"] | None = None
    source_channel_names: list[str] | None = None
    canonical_channel_names: list[str] | None = None
    name_sources: list[str] | None = None
    ome_tiff: OmeTiffRecord | None = None
    source: SourceRecord | None = None
    czi: CziRecord | None = None
    # result-store fields
    full_shape_yx: tuple[int, int] | None = None
    crop: CropRect | None = None
    source_shape_yx: tuple[int, int] | None = None
    source_pixel_size_um: float | None = None
    source_crop: CropRect | None = None

    @field_validator("pixel_size_um", "source_pixel_size_um")
    @classmethod
    def _finite_pixel_size(
        cls: type[StoreImageRecord], value: float | None
    ) -> float | None:
        return _finite(value)

    @model_validator(mode="after")
    def _validate_registration_channel(
        self: StoreImageRecord,
    ) -> StoreImageRecord:
        if self.registration_channel not in self.channel_names:
            raise ValueError(
                f"registration_channel {self.registration_channel!r} is not one of "
                f"the channel names {self.channel_names}"
            )
        return self


class ChannelMatching(BaseModel):
    """Which reference channels were kept to match the moving image."""

    model_config = ConfigDict(extra="ignore")

    reference_kept: list[str]
    reference_dropped: list[str]
    moving_unmatched: list[str]


class PreparedStoreRef(BaseModel):
    """Reference from a result store to the prepared store it was made from."""

    model_config = ConfigDict(extra="ignore")

    path: str
    content_digest: str = Field(pattern=DIGEST_PATTERN)
    grid_digest: str = Field(pattern=DIGEST_PATTERN)


class StoreAttrs(BaseModel):
    """Root attrs ``STORE_ATTR`` of a prepared or result store."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    schema_: Literal["merxen.manual_registration.store"] = Field(alias="schema")
    schema_version: int
    kind: Literal["prepared", "result"]
    dataset_name: str = Field(pattern=DATASET_ID_PATTERN)
    created_at: str
    merxen_version: str
    conventions: dict[str, str]
    images: dict[str, StoreImageRecord]
    channel_matching: ChannelMatching | None = None
    grid_digest: str | None = Field(default=None, pattern=DIGEST_PATTERN)
    content_digest: str | None = Field(default=None, pattern=DIGEST_PATTERN)
    run_id: str | None = Field(default=None, pattern=DATASET_ID_PATTERN)
    preset: str | None = None
    run_label: str | None = None
    prepared_store: PreparedStoreRef | None = None

    @field_validator("schema_version")
    @classmethod
    def _check_version(cls: type[StoreAttrs], value: int) -> int:
        check_schema_version({"schema_version": value}, attr=STORE_ATTR)
        return value

    @model_validator(mode="after")
    def _validate_store(self: StoreAttrs) -> StoreAttrs:
        if set(self.images) != {"reference", "moving"}:
            raise ValueError("images must have exactly 'reference' and 'moving'")
        for key, record in self.images.items():
            if record.role != key:
                raise ValueError(
                    f"images[{key!r}].role must be {key!r}, got {record.role!r}"
                )
        if self.kind == "prepared" and (
            self.grid_digest is None
            or self.content_digest is None
            or self.channel_matching is None
        ):
            raise ValueError(
                "a prepared store needs grid_digest, content_digest and "
                "channel_matching"
            )
        if self.kind == "result" and (
            self.run_id is None or self.preset is None or self.prepared_store is None
        ):
            raise ValueError("a result store needs run_id, preset and prepared_store")
        return self


def parse_labelling_attrs(payload: Mapping[str, Any]) -> LabellingAttrs:
    """Validate a ``LABELLING_ATTR`` payload.

    Raises:
        UnsupportedSchemaVersionError: If the payload is from a newer writer.
        ValueError: If the payload is not valid (pydantic ``ValidationError``).
    """
    check_schema_version(payload, attr=LABELLING_ATTR)
    return LabellingAttrs.model_validate(dict(payload))


def parse_store_attrs(payload: Mapping[str, Any]) -> StoreAttrs:
    """Validate a ``STORE_ATTR`` payload.

    Raises:
        UnsupportedSchemaVersionError: If the payload is from a newer writer.
        ValueError: If the payload is not valid (pydantic ``ValidationError``).
    """
    check_schema_version(payload, attr=STORE_ATTR)
    return StoreAttrs.model_validate(dict(payload))


def canonical_json(obj: Any) -> str:
    """Serialise plain JSON data canonically (sorted keys, no spaces, ASCII).

    Raises:
        ValueError: If the data contains NaN or an infinity.
    """
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def canonical_sha256(obj: Any) -> str:
    """Return ``sha256:<hex>`` of the canonical JSON of ``obj``."""
    return "sha256:" + hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def labelling_content_sha256(payload: Mapping[str, Any]) -> str:
    """Digest of the user-authored content of a labelling payload.

    Covers the coarse state, the crops and the landmark pairs, and nothing else
    (not the revision, timestamps, status or the informative matrix). It must be
    computed on the raw mapping read from attrs, never on a pydantic dump:
    pydantic turns ``8811`` into ``8811.0`` and tuples into lists.

    Raises:
        ValueError: If the payload lacks a needed key.
    """
    try:
        coarse = payload["coarse"]
        content = {
            "coarse": {
                "rotation_deg": coarse["rotation_deg"],
                "mirror_x": coarse["mirror_x"],
                "offset_um": coarse["offset_um"],
            },
            "crops": payload["crops"],
            "pairs": payload["landmarks"]["pairs"],
        }
    except (KeyError, TypeError) as exc:
        key = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        raise ValueError(f"labelling payload lacks {key!r}") from exc
    return canonical_sha256(content)


def file_fingerprint(path: Path, *, chunk_bytes: int = 16 * 2**20) -> str:
    """Fingerprint a file by its size and the hashes of its first and last bytes.

    Independent of the file's path and modification time.

    Args:
        path: The file to fingerprint.
        chunk_bytes: How many bytes to hash at each end.

    Raises:
        ValueError: If ``chunk_bytes`` is less than 1.
    """
    if chunk_bytes < 1:
        raise ValueError(f"chunk_bytes must be at least 1, got {chunk_bytes}")
    size = path.stat().st_size
    window = min(size, chunk_bytes)
    with path.open("rb") as handle:
        head = hashlib.sha256(handle.read(window)).hexdigest()
        handle.seek(max(0, size - chunk_bytes))
        tail = hashlib.sha256(handle.read(window)).hexdigest()
    return canonical_sha256({"size": size, "head": head, "tail": tail})


def pair_colour(pair_id: int, seed: int) -> str:
    """Return the deterministic ``#rrggbb`` colour of a landmark pair."""
    base_hue = ((seed * 2654435761) % 2**32) / 2**32
    hue = (base_hue + _GOLDEN_RATIO_CONJUGATE * pair_id) % 1.0
    saturation = (0.75, 1.0, 0.9)[pair_id % 3]
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, 1.0)
    return f"#{round(255 * red):02x}{round(255 * green):02x}{round(255 * blue):02x}"


def utc_now() -> str:
    """Return the current UTC time as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_root_attrs(store: Path) -> dict[str, Any]:
    """Read the root attributes of a Zarr group without zarr or a lock.

    Handles Zarr v3 (``zarr.json``, authoritative even when consolidated
    metadata is present) and v2 (``.zgroup`` / ``.zattrs``) layouts.

    Raises:
        FileNotFoundError: If ``store`` is not a Zarr group.
        ValueError: If ``zarr.json`` describes an array.
    """
    store = Path(store)
    v3_path = store / "zarr.json"
    if v3_path.exists():
        raw = json.loads(v3_path.read_text(encoding="utf-8"))
        if raw.get("node_type") != "group":
            raise ValueError(f"{store} is not a Zarr group")
        return dict(raw.get("attributes", {}))
    if (store / ".zgroup").exists():
        attrs_path = store / ".zattrs"
        if attrs_path.exists():
            return dict(json.loads(attrs_path.read_text(encoding="utf-8")))
        return {}
    raise FileNotFoundError(f"{store} is not a Zarr group (no zarr.json or .zgroup)")


def read_root_attrs_locked(store: Path) -> dict[str, Any]:
    """Read the root attributes while holding the store's write lock.

    Waits for a concurrent writer. The lock is not re-entrant: do not call this
    from a thread that already holds ``spatialdata_write_lock(store)`` (it raises
    ``RuntimeError``); use ``read_root_attrs`` there.
    """
    from merxen.io.spatialdata_io import spatialdata_write_lock

    with spatialdata_write_lock(store):
        return read_root_attrs(store)


def write_root_attr(store: Path, key: str, payload: Mapping[str, Any]) -> None:
    """Set one root attribute of an existing Zarr group, keeping all others.

    The caller must already hold ``spatialdata_write_lock(store)``. The store is
    never created, other root keys and the consolidated metadata are kept, and
    the value is read back and compared.

    Args:
        store: Existing Zarr group (for example a SpatialData store).
        key: Root attribute name.
        payload: Plain JSON data (no numpy scalars, no NaN).

    Raises:
        RuntimeError: If the lock is not held or the read-back differs.
        FileNotFoundError: If ``store`` is not a Zarr group.
        ValueError: If the payload is not plain JSON.
    """
    # Private helper: the lock registry is the only way to check the contract.
    from merxen.io.spatialdata_io import (
        _write_locks_held_by_this_thread,
        spatialdata_write_lock_path,
    )

    store = Path(store)
    if spatialdata_write_lock_path(store) not in _write_locks_held_by_this_thread():
        raise RuntimeError(
            f"write_root_attr needs the caller to hold spatialdata_write_lock({store})"
        )
    read_root_attrs(store)  # raises if the store is not an existing group
    try:
        plain = json.loads(canonical_json(payload))
    except TypeError as exc:
        raise ValueError(f"{key}: payload is not plain JSON: {exc}") from exc

    import zarr

    # Default use_consolidated: passing False would drop the root
    # consolidated_metadata on write.
    group = zarr.open_group(str(store), mode="a")
    group.attrs.update({key: plain})
    if read_root_attrs(store).get(key) != plain:
        raise RuntimeError(f"{key}: read-back after write differs")
