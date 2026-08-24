"""Glyph rasterization with a freetype-py-compatible Python surface."""

from __future__ import annotations

import math

import freetype as _ft

from ._lib import _addr_unchecked
from .raster import (
    Bitmap,
    DEFAULT_CONFIG,
    RasterConfig,
    _Scratch,
    _prepare_row_edges,
    _prepare_segments,
    flatten_outline,
    rasterize,
)

for _name in dir(_ft):
    if not _name.startswith("_") and _name not in globals():
        globals()[_name] = getattr(_ft, _name)


class GlyphSlot:
    """Proxy for ``freetype.GlyphSlot`` with Mojo-backed ``render``."""

    def __init__(self, slot):
        self._slot = slot
        self._bitmap = None
        self._bitmap_left = None
        self._bitmap_top = None

    def __getattr__(self, name):
        return getattr(self._slot, name)

    @property
    def bitmap(self):
        return self._bitmap if self._bitmap is not None else self._slot.bitmap

    @property
    def bitmap_left(self):
        return (
            self._bitmap_left
            if self._bitmap_left is not None
            else self._slot.bitmap_left
        )

    @property
    def bitmap_top(self):
        return (
            self._bitmap_top
            if self._bitmap_top is not None
            else self._slot.bitmap_top
        )

    def render(
        self,
        render_mode,
        *,
        _segments=None,
        _segments_addr=0,
        _bounds=None,
        _even_odd=None,
        _scratch=None,
        _row_offsets=None,
        _row_edges=None,
        _row_offsets_addr=0,
        _row_edges_addr=0,
    ):
        bitmap, left, top = rasterize(
            self._slot.outline,
            render_mode,
            _segments=_segments,
            _segments_addr=_segments_addr,
            _bounds=_bounds,
            _even_odd=_even_odd,
            _scratch=_scratch,
            _row_offsets=_row_offsets,
            _row_edges=_row_edges,
            _row_offsets_addr=_row_offsets_addr,
            _row_edges_addr=_row_edges_addr,
        )
        self._bitmap = bitmap
        self._bitmap_left = left
        self._bitmap_top = top


def _target_mode(flags: int) -> int:
    return (int(flags) >> 16) & 15


class Face:
    """A ``freetype.Face`` proxy whose requested outline rendering uses Mojo."""

    def __init__(self, path_or_stream, index=0):
        self._face = _ft.Face(path_or_stream, index)
        self._glyph = GlyphSlot(self._face.glyph)
        self._segment_cache = {}
        self._scratch = _Scratch()
        self._state_generation = 0

    def __getattr__(self, name):
        return getattr(self._face, name)

    @property
    def glyph(self) -> GlyphSlot:
        return self._glyph

    def _invalidate_segments(self) -> None:
        self._state_generation += 1
        self._segment_cache.clear()

    def set_pixel_sizes(self, width, height):
        result = self._face.set_pixel_sizes(width, height)
        self._invalidate_segments()
        return result

    def set_char_size(self, width=0, height=0, hres=72, vres=72):
        result = self._face.set_char_size(width, height, hres, vres)
        self._invalidate_segments()
        return result

    def set_transform(self, matrix, delta):
        result = self._face.set_transform(matrix, delta)
        self._invalidate_segments()
        return result

    def select_size(self, strike_index):
        result = self._face.select_size(strike_index)
        self._invalidate_segments()
        return result

    def select_charmap(self, encoding):
        result = self._face.select_charmap(encoding)
        self._invalidate_segments()
        return result

    def set_charmap(self, charmap):
        result = self._face.set_charmap(charmap)
        self._invalidate_segments()
        return result

    def set_var_blend_coords(self, coords, reset=False):
        result = self._face.set_var_blend_coords(coords, reset)
        self._invalidate_segments()
        return result

    def set_var_design_coords(self, coords, reset=False):
        result = self._face.set_var_design_coords(coords, reset)
        self._invalidate_segments()
        return result

    def set_var_named_instance(self, instance_index):
        result = self._face.set_var_named_instance(instance_index)
        self._invalidate_segments()
        return result

    def _finish_load(self, flags: int, glyph_index: int) -> None:
        self._glyph = GlyphSlot(self._face.glyph)
        if flags & _ft.FT_LOAD_RENDER:
            if self._glyph.format != _ft.FT_GLYPH_FORMAT_OUTLINE:
                raise NotImplementedError(
                    "Mojo rendering requires a scalable outline glyph"
                )
            mode = _target_mode(flags)
            if mode not in (
                _ft.FT_RENDER_MODE_NORMAL,
                _ft.FT_RENDER_MODE_LIGHT,
                _ft.FT_RENDER_MODE_MONO,
            ):
                raise NotImplementedError(
                    "Mojo rendering covers normal, light, and mono targets"
                )
            cache_key = (
                self._state_generation,
                int(glyph_index),
                int(flags) & ~_ft.FT_LOAD_RENDER,
            )
            prepared = self._segment_cache.get(cache_key)
            if prepared is None:
                outline = self._glyph._slot.outline
                segments = _prepare_segments(
                    flatten_outline(outline)
                )
                segments_addr = _addr_unchecked(segments)
                bounds = None
                if outline.n_points:
                    bbox = outline.get_bbox()
                    bounds = (
                        math.floor(bbox.xMin / 64.0),
                        math.ceil(bbox.xMax / 64.0),
                        math.floor(bbox.yMin / 64.0),
                        math.ceil(bbox.yMax / 64.0),
                    )
                even_odd = int(
                    bool(outline.flags & _ft.FT_OUTLINE_EVEN_ODD_FILL)
                )
                row_offsets = None
                row_edges = None
                if (
                    bounds is not None
                    and segments.shape[1]
                    and (bounds[3] - bounds[2]) * segments.shape[1] >= 512
                ):
                    row_offsets, row_edges = _prepare_row_edges(
                        segments, bounds[3], bounds[3] - bounds[2]
                    )
                row_offsets_addr = (
                    0
                    if row_offsets is None
                    else _addr_unchecked(row_offsets)
                )
                row_edges_addr = (
                    0 if row_edges is None else _addr_unchecked(row_edges)
                )
                prepared = (
                    segments,
                    segments_addr,
                    bounds,
                    even_odd,
                    row_offsets,
                    row_edges,
                    row_offsets_addr,
                    row_edges_addr,
                )
                self._segment_cache[cache_key] = prepared
            (
                segments,
                segments_addr,
                bounds,
                even_odd,
                row_offsets,
                row_edges,
                row_offsets_addr,
                row_edges_addr,
            ) = prepared
            self._glyph.render(
                mode,
                _segments=segments,
                _segments_addr=segments_addr,
                _bounds=bounds,
                _even_odd=even_odd,
                _scratch=self._scratch,
                _row_offsets=row_offsets,
                _row_edges=row_edges,
                _row_offsets_addr=row_offsets_addr,
                _row_edges_addr=row_edges_addr,
            )

    def load_char(self, char, flags=_ft.FT_LOAD_RENDER):
        load_flags = int(flags)
        if load_flags & _ft.FT_LOAD_RENDER:
            load_flags = (
                (load_flags & ~_ft.FT_LOAD_RENDER) | _ft.FT_LOAD_NO_BITMAP
            )
        self._face.load_char(char, load_flags)
        self._finish_load(int(flags), self._face.get_char_index(char))

    def load_glyph(self, index, flags=_ft.FT_LOAD_RENDER):
        load_flags = int(flags)
        if load_flags & _ft.FT_LOAD_RENDER:
            load_flags = (
                (load_flags & ~_ft.FT_LOAD_RENDER) | _ft.FT_LOAD_NO_BITMAP
            )
        self._face.load_glyph(index, load_flags)
        self._finish_load(int(flags), int(index))


__all__ = sorted(
    {
        *(name for name in dir(_ft) if not name.startswith("_")),
        "Bitmap",
        "Face",
        "GlyphSlot",
        "RasterConfig",
        "DEFAULT_CONFIG",
        "flatten_outline",
        "rasterize",
    }
)
