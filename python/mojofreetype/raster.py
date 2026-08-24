"""FreeType-compatible outline rasterization backed by Mojo."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import Callable

import freetype as _ft
import numpy as np

from ._lib import _addr_unchecked, addr, lib


@dataclass(frozen=True)
class RasterConfig:
    samples: int = 12
    curve_tolerance: float = 1.0 / 8.0
    max_curve_depth: int = 12
    parallel_threshold: int = 10_000_000


DEFAULT_CONFIG = RasterConfig()
_INT64_MAX = (1 << 63) - 1


class _Scratch:
    def __init__(self) -> None:
        self.intersections = np.empty(0, dtype=np.float64)
        self.directions = np.empty(0, dtype=np.int32)
        self.active_edges = np.empty(0, dtype=np.int32)
        self.coverage = np.empty(0, dtype=np.float64)
        self.intersections_addr = 0
        self.directions_addr = 0
        self.active_edges_addr = 0
        self.coverage_addr = 0
        self.edge_capacity = 0
        self.coverage_capacity = 0

    def reserve(self, rows: int, edge_count: int, width: int) -> None:
        edge_size = rows * max(1, edge_count)
        if edge_size > self.edge_capacity:
            self.edge_capacity = max(edge_size, self.edge_capacity * 2)
            self.intersections = np.empty(
                self.edge_capacity, dtype=np.float64
            )
            self.directions = np.empty(self.edge_capacity, dtype=np.int32)
            self.active_edges = np.empty(
                self.edge_capacity, dtype=np.int32
            )
            self.intersections_addr = _addr_unchecked(self.intersections)
            self.directions_addr = _addr_unchecked(self.directions)
            self.active_edges_addr = _addr_unchecked(self.active_edges)
        coverage_size = rows * width
        if coverage_size > self.coverage_capacity:
            self.coverage_capacity = max(
                coverage_size, self.coverage_capacity * 2
            )
            self.coverage = np.empty(
                self.coverage_capacity, dtype=np.float64
            )
            self.coverage_addr = _addr_unchecked(self.coverage)


def _validate_config(config: RasterConfig) -> None:
    for name in ("samples", "max_curve_depth", "parallel_threshold"):
        value = getattr(config, name)
        try:
            integer = operator.index(value)
        except TypeError as error:
            raise TypeError(f"{name} must be an integer") from error
        if integer < (1 if name == "samples" else 0):
            qualifier = "positive" if name == "samples" else "non-negative"
            raise ValueError(f"{name} must be {qualifier}")
        if integer > _INT64_MAX:
            raise OverflowError(f"{name} does not fit the Mojo C ABI")
    if not math.isfinite(config.curve_tolerance) or config.curve_tolerance <= 0:
        raise ValueError("curve_tolerance must be finite and positive")


class Bitmap:
    """The covered subset of :class:`freetype.Bitmap`."""

    def __init__(self, array: np.ndarray, pixel_mode: int = _ft.FT_PIXEL_MODE_GRAY):
        if (
            isinstance(array, np.ndarray)
            and array.dtype == np.uint8
            and array.flags.c_contiguous
        ):
            self._array = array
        else:
            self._array = np.ascontiguousarray(array, dtype=np.uint8)
        self._pixel_mode = int(pixel_mode)

    @property
    def rows(self) -> int:
        return int(self._array.shape[0])

    @property
    def width(self) -> int:
        if self.pixel_mode == _ft.FT_PIXEL_MODE_MONO:
            return int(self._width)
        return int(self._array.shape[1]) if self._array.ndim == 2 else 0

    @property
    def pitch(self) -> int:
        return int(self._array.shape[1]) if self._array.ndim == 2 else 0

    @property
    def buffer(self) -> list[int]:
        return self._array.ravel().tolist()

    @property
    def num_grays(self) -> int:
        return 256 if self.pixel_mode == _ft.FT_PIXEL_MODE_GRAY else 2

    @property
    def pixel_mode(self) -> int:
        return self._pixel_mode

    @property
    def palette_mode(self) -> int:
        return 0

    @property
    def palette(self):
        return None

    def to_array(self, unpack: bool = True) -> np.ndarray:
        """Return a copy of the coverage image, unpacking mono bits by default."""
        if self.pixel_mode != _ft.FT_PIXEL_MODE_MONO or not unpack:
            return self._array.copy()
        bits = np.unpackbits(self._array, axis=1, bitorder="big")
        return (bits[:, : self.width] * 255).astype(np.uint8)

    @classmethod
    def _mono(cls, array: np.ndarray, width: int) -> "Bitmap":
        bitmap = cls(array, _ft.FT_PIXEL_MODE_MONO)
        bitmap._width = int(width)
        return bitmap


def _flat_enough(
    p0: tuple[float, float],
    controls: tuple[tuple[float, float], ...],
    p1: tuple[float, float],
    tolerance: float,
) -> bool:
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    chord2 = dx * dx + dy * dy
    if chord2 == 0.0:
        return max(
            (point[0] - p0[0]) ** 2 + (point[1] - p0[1]) ** 2
            for point in controls
        ) <= tolerance * tolerance
    limit = tolerance * tolerance * chord2
    return all(
        (dx * (point[1] - p0[1]) - dy * (point[0] - p0[0])) ** 2 <= limit
        for point in controls
    )


def _flatten_quadratic(
    p0: tuple[float, float],
    control: tuple[float, float],
    p1: tuple[float, float],
    emit: Callable[[tuple[float, float], tuple[float, float]], None],
    tolerance: float,
    depth: int,
) -> None:
    if depth == 0 or _flat_enough(p0, (control,), p1, tolerance):
        emit(p0, p1)
        return
    p01 = ((p0[0] + control[0]) * 0.5, (p0[1] + control[1]) * 0.5)
    p12 = ((control[0] + p1[0]) * 0.5, (control[1] + p1[1]) * 0.5)
    midpoint = ((p01[0] + p12[0]) * 0.5, (p01[1] + p12[1]) * 0.5)
    _flatten_quadratic(p0, p01, midpoint, emit, tolerance, depth - 1)
    _flatten_quadratic(midpoint, p12, p1, emit, tolerance, depth - 1)


def _flatten_cubic(
    p0: tuple[float, float],
    c1: tuple[float, float],
    c2: tuple[float, float],
    p1: tuple[float, float],
    emit: Callable[[tuple[float, float], tuple[float, float]], None],
    tolerance: float,
    depth: int,
) -> None:
    if depth == 0 or _flat_enough(p0, (c1, c2), p1, tolerance):
        emit(p0, p1)
        return
    p01 = ((p0[0] + c1[0]) * 0.5, (p0[1] + c1[1]) * 0.5)
    p12 = ((c1[0] + c2[0]) * 0.5, (c1[1] + c2[1]) * 0.5)
    p23 = ((c2[0] + p1[0]) * 0.5, (c2[1] + p1[1]) * 0.5)
    p012 = ((p01[0] + p12[0]) * 0.5, (p01[1] + p12[1]) * 0.5)
    p123 = ((p12[0] + p23[0]) * 0.5, (p12[1] + p23[1]) * 0.5)
    midpoint = ((p012[0] + p123[0]) * 0.5, (p012[1] + p123[1]) * 0.5)
    _flatten_cubic(p0, p01, p012, midpoint, emit, tolerance, depth - 1)
    _flatten_cubic(midpoint, p123, p23, p1, emit, tolerance, depth - 1)


def flatten_outline(
    outline,
    origin: tuple[float, float] = (0.0, 0.0),
    config: RasterConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """Convert a freetype-py ``Outline`` into ``(x0, y0, x1, y1)`` edges."""
    if config is not DEFAULT_CONFIG:
        _validate_config(config)
    edges: list[tuple[float, float, float, float]] = []
    start: tuple[float, float] | None = None
    current: tuple[float, float] | None = None
    ox, oy = origin

    def point(vector) -> tuple[float, float]:
        return (vector.x / 64.0 + ox, vector.y / 64.0 + oy)

    def emit(a: tuple[float, float], b: tuple[float, float]) -> None:
        if a != b:
            edges.append((a[0], a[1], b[0], b[1]))

    def close() -> None:
        nonlocal current
        if current is not None and start is not None:
            emit(current, start)
            current = start

    def move_to(vector, _context) -> None:
        nonlocal start, current
        close()
        start = current = point(vector)

    def line_to(vector, _context) -> None:
        nonlocal current
        target = point(vector)
        emit(current, target)
        current = target

    def conic_to(control, vector, _context) -> None:
        nonlocal current
        target = point(vector)
        _flatten_quadratic(
            current,
            point(control),
            target,
            emit,
            config.curve_tolerance,
            config.max_curve_depth,
        )
        current = target

    def cubic_to(c1, c2, vector, _context) -> None:
        nonlocal current
        target = point(vector)
        _flatten_cubic(
            current,
            point(c1),
            point(c2),
            target,
            emit,
            config.curve_tolerance,
            config.max_curve_depth,
        )
        current = target

    outline.decompose(
        move_to=move_to,
        line_to=line_to,
        conic_to=conic_to,
        cubic_to=cubic_to,
    )
    close()
    return np.ascontiguousarray(edges, dtype=np.float64).reshape((-1, 4))


def _prepare_segments(segments: np.ndarray) -> np.ndarray:
    segments = segments[segments[:, 1] != segments[:, 3]]
    count = len(segments)
    prepared = np.empty((5, count), dtype=np.float64)
    if count == 0:
        return prepared
    upward = segments[:, 3] > segments[:, 1]
    prepared[0] = np.where(upward, segments[:, 0], segments[:, 2])
    prepared[1] = np.minimum(segments[:, 1], segments[:, 3])
    prepared[2] = np.maximum(segments[:, 1], segments[:, 3])
    prepared[3] = (
        (segments[:, 2] - segments[:, 0])
        / (segments[:, 3] - segments[:, 1])
    )
    prepared[4] = np.where(upward, 1.0, -1.0)
    return prepared


def _prepare_row_edges(
    segments: np.ndarray, top: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.empty(height + 1, dtype=np.int32)
    offsets[0] = 0
    rows: list[np.ndarray] = []
    total = 0
    for row in range(height):
        row_top = top - row
        active = np.flatnonzero(
            (segments[1] < row_top) & (segments[2] > row_top - 1)
        ).astype(np.int32, copy=False)
        rows.append(active)
        total += active.size
        offsets[row + 1] = total
    edges = (
        np.concatenate(rows)
        if total
        else np.empty(0, dtype=np.int32)
    )
    return offsets, edges


def rasterize(
    outline,
    render_mode: int = _ft.FT_RENDER_MODE_NORMAL,
    origin=None,
    config: RasterConfig = DEFAULT_CONFIG,
    *,
    _segments: np.ndarray | None = None,
    _segments_addr: int = 0,
    _bounds: tuple[int, int, int, int] | None = None,
    _even_odd: int | None = None,
    _scratch: _Scratch | None = None,
    _row_offsets: np.ndarray | None = None,
    _row_edges: np.ndarray | None = None,
    _row_offsets_addr: int = 0,
    _row_edges_addr: int = 0,
) -> tuple[Bitmap, int, int]:
    """Rasterize an upstream ``Outline`` and return ``(bitmap, left, top)``."""
    if config is not DEFAULT_CONFIG:
        _validate_config(config)
    if render_mode not in (
        _ft.FT_RENDER_MODE_NORMAL,
        _ft.FT_RENDER_MODE_LIGHT,
        _ft.FT_RENDER_MODE_MONO,
    ):
        raise NotImplementedError(
            "only FT_RENDER_MODE_NORMAL, LIGHT, and MONO are covered"
        )
    if _bounds is None and outline.n_points == 0:
        empty = np.empty((0, 0), dtype=np.uint8)
        bitmap = (
            Bitmap._mono(empty, 0)
            if render_mode == _ft.FT_RENDER_MODE_MONO
            else Bitmap(empty)
        )
        return bitmap, 0, 0
    if origin is None:
        translation = (0.0, 0.0)
    elif hasattr(origin, "x") and hasattr(origin, "y"):
        translation = (origin.x / 64.0, origin.y / 64.0)
    else:
        translation = (float(origin[0]), float(origin[1]))

    if _bounds is None:
        bbox = outline.get_bbox()
        left = math.floor(bbox.xMin / 64.0 + translation[0])
        right = math.ceil(bbox.xMax / 64.0 + translation[0])
        bottom = math.floor(bbox.yMin / 64.0 + translation[1])
        top = math.ceil(bbox.yMax / 64.0 + translation[1])
    else:
        left, right, bottom, top = _bounds
    width, height = max(0, right - left), max(0, top - bottom)
    if width == 0 or height == 0:
        empty = np.empty((height, width), dtype=np.uint8)
        bitmap = (
            Bitmap._mono(empty, width)
            if render_mode == _ft.FT_RENDER_MODE_MONO
            else Bitmap(empty)
        )
        return bitmap, left, top

    if _segments is None:
        segments = _prepare_segments(
            flatten_outline(outline, translation, config)
        )
    else:
        if not isinstance(_segments, np.ndarray):
            raise TypeError("_segments must be a NumPy array")
        if (
            _segments.dtype != np.float64
            or _segments.ndim != 2
            or _segments.shape[0] != 5
            or not _segments.flags.c_contiguous
        ):
            raise ValueError(
                "_segments must be a C-contiguous float64 array shaped (5, n)"
            )
        segments = _segments
    edge_count = segments.shape[1]
    gray = np.empty((height, width), dtype=np.uint8)
    if edge_count == 0:
        gray.fill(0)
        if render_mode in (
            _ft.FT_RENDER_MODE_NORMAL,
            _ft.FT_RENDER_MODE_LIGHT,
        ):
            return Bitmap(gray), left, top
        mono_pitch = ((width + 15) // 16) * 2
        return Bitmap._mono(
            np.zeros((height, mono_pitch), dtype=np.uint8), width
        ), left, top
    parallel = (
        height >= 64
        and height * edge_count * config.samples
        >= config.parallel_threshold
    )
    scratch_rows = 1
    scratch = _Scratch() if _scratch is None else _scratch
    scratch.reserve(scratch_rows, edge_count, width)
    intersections = scratch.intersections
    directions = scratch.directions
    active_edges = scratch.active_edges
    coverage = scratch.coverage
    even_odd = (
        int(bool(outline.flags & _ft.FT_OUTLINE_EVEN_ODD_FILL))
        if _even_odd is None
        else _even_odd
    )
    lib().mft_raster_gray(
        _segments_addr or _addr_unchecked(segments),
        edge_count,
        scratch.intersections_addr,
        scratch.directions_addr,
        scratch.active_edges_addr,
        (
            0
            if _row_offsets is None
            else _row_offsets_addr or _addr_unchecked(_row_offsets)
        ),
        (
            0
            if _row_edges is None
            else _row_edges_addr or _addr_unchecked(_row_edges)
        ),
        scratch.coverage_addr,
        _addr_unchecked(gray),
        width,
        height,
        width,
        left,
        top,
        config.samples,
        even_odd,
        int(parallel),
    )
    if render_mode in (_ft.FT_RENDER_MODE_NORMAL, _ft.FT_RENDER_MODE_LIGHT):
        return Bitmap(gray), left, top

    mono_pitch = ((width + 15) // 16) * 2
    packed = np.empty((height, mono_pitch), dtype=np.uint8)
    lib().mft_pack_mono(
        _addr_unchecked(gray),
        _addr_unchecked(packed),
        width,
        height,
        width,
        mono_pitch,
    )
    return Bitmap._mono(packed, width), left, top
