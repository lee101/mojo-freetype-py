from __future__ import annotations

import io
import subprocess
from pathlib import Path
from types import SimpleNamespace

import freetype
import numpy as np
import pytest

import mojofreetype as mft
from mojofreetype._lib import addr, lib


def _font() -> str:
    return _matching_font("DejaVu Sans")


def _matching_font(pattern: str) -> str:
    result = subprocess.run(
        ["fc-match", "-f", "%{file}", pattern],
        check=True,
        capture_output=True,
        text=True,
    )
    path = result.stdout.strip()
    if not path or not Path(path).is_file():
        pytest.skip(f"no font matching {pattern!r} installed")
    return path


def _upstream(char: str, size: int, mode: int = freetype.FT_RENDER_MODE_NORMAL):
    face = freetype.Face(_font())
    face.set_pixel_sizes(0, size)
    flags = freetype.FT_LOAD_DEFAULT
    if mode == freetype.FT_RENDER_MODE_MONO:
        flags |= freetype.FT_LOAD_TARGET_MONO
    face.load_char(char, flags)
    face.glyph.render(mode)
    bitmap = face.glyph.bitmap
    raw = np.asarray(bitmap.buffer, dtype=np.uint8).reshape(
        bitmap.rows, bitmap.pitch
    )
    if mode == freetype.FT_RENDER_MODE_MONO:
        raw = (
            np.unpackbits(raw, axis=1, bitorder="big")[:, : bitmap.width] * 255
        ).astype(np.uint8)
    else:
        raw = raw[:, : bitmap.width]
    return raw, face.glyph.bitmap_left, face.glyph.bitmap_top


def _ours(char: str, size: int, mode: int = freetype.FT_RENDER_MODE_NORMAL):
    face = mft.Face(_font())
    face.set_pixel_sizes(0, size)
    flags = freetype.FT_LOAD_RENDER
    if mode == freetype.FT_RENDER_MODE_MONO:
        flags |= freetype.FT_LOAD_TARGET_MONO
    face.load_char(char, flags)
    return (
        face.glyph.bitmap.to_array(),
        face.glyph.bitmap_left,
        face.glyph.bitmap_top,
    )


@pytest.mark.parametrize("char", ["A", "g", "@", "é", "Ж", "8"])
@pytest.mark.parametrize("size", [12, 32, 72])
def test_gray_render_parity(char, size):
    ours, left, top = _ours(char, size)
    reference, ref_left, ref_top = _upstream(char, size)
    assert (left, top, ours.shape) == (ref_left, ref_top, reference.shape)
    difference = np.abs(ours.astype(np.int16) - reference.astype(np.int16))
    assert difference.mean() < 3.0
    assert np.quantile(difference, 0.99) <= 24
    ours_ink = ours > 8
    reference_ink = reference > 8
    union = np.count_nonzero(ours_ink | reference_ink)
    assert np.count_nonzero(ours_ink & reference_ink) / union > 0.96


@pytest.mark.parametrize("char", ["A", "g", "@", "é"])
@pytest.mark.parametrize("size", [16, 48])
def test_mono_render_behavioral_parity(char, size):
    ours, left, top = _ours(char, size, freetype.FT_RENDER_MODE_MONO)
    reference, ref_left, ref_top = _upstream(
        char, size, freetype.FT_RENDER_MODE_MONO
    )
    assert (left, top, ours.shape) == (ref_left, ref_top, reference.shape)
    assert set(np.unique(ours)) <= {0, 255}
    agreement = np.mean(ours == reference)
    assert agreement > 0.97


def test_load_glyph_matches_load_char():
    face = mft.Face(_font())
    face.set_pixel_sizes(0, 40)
    index = face.get_char_index("Q")
    face.load_glyph(index)
    by_index = face.glyph.bitmap.to_array()
    face.load_char("Q")
    assert np.array_equal(by_index, face.glyph.bitmap.to_array())


def test_face_accepts_font_stream_and_proxies_metrics():
    font_data = Path(_font()).read_bytes()
    face = mft.Face(io.BytesIO(font_data))
    face.set_pixel_sizes(0, 30)
    face.load_char("M")
    assert face.glyph.metrics.width > 0
    assert face.glyph.advance.x > 0
    assert face.glyph.bitmap.width > 0


def test_load_without_render_preserves_native_outline():
    face = mft.Face(_font())
    reference = freetype.Face(_font())
    face.set_pixel_sizes(0, 40)
    reference.set_pixel_sizes(0, 40)
    face.load_char("S", freetype.FT_LOAD_DEFAULT)
    reference.load_char("S", freetype.FT_LOAD_DEFAULT)
    assert face.glyph.outline.n_points > 0
    assert isinstance(face.glyph.bitmap, freetype.Bitmap)
    assert face.glyph.format == reference.glyph.format
    assert face.glyph.outline.points == reference.glyph.outline.points


def test_glyphslot_render_signature_and_mutation():
    face = mft.Face(_font())
    face.set_pixel_sizes(0, 40)
    face.load_char("B", freetype.FT_LOAD_DEFAULT)
    assert face.glyph.render(freetype.FT_RENDER_MODE_NORMAL) is None
    assert isinstance(face.glyph.bitmap, mft.Bitmap)
    assert face.glyph.bitmap.num_grays == 256
    assert face.glyph.bitmap.pixel_mode == freetype.FT_PIXEL_MODE_GRAY


def test_direct_rasterize_and_bitmap_contract():
    face = freetype.Face(_font())
    face.set_pixel_sizes(0, 36)
    face.load_char("&", freetype.FT_LOAD_DEFAULT)
    bitmap, left, top = mft.rasterize(face.glyph.outline)
    assert bitmap.rows > 0 and bitmap.width > 0
    assert bitmap.pitch == bitmap.width
    assert len(bitmap.buffer) == bitmap.rows * bitmap.pitch
    assert bitmap.palette is None and bitmap.palette_mode == 0
    assert isinstance(left, int) and isinstance(top, int)


def test_direct_rasterize_origin_translates_bounds():
    face = freetype.Face(_font())
    face.set_pixel_sizes(0, 36)
    face.load_char("A", freetype.FT_LOAD_DEFAULT)
    plain, left, top = mft.rasterize(face.glyph.outline)
    shifted, shifted_left, shifted_top = mft.rasterize(
        face.glyph.outline, origin=(2.0, -3.0)
    )
    assert (shifted_left, shifted_top) == (left + 2, top - 3)
    assert np.array_equal(shifted.to_array(), plain.to_array())


def test_empty_space_glyph():
    ours, left, top = _ours(" ", 32)
    assert ours.shape == (0, 0)
    assert (left, top) == (0, 0)
    face = mft.Face(_font())
    face.set_pixel_sizes(0, 32)
    face.load_char(
        " ", freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_MONO
    )
    assert face.glyph.bitmap.pixel_mode == freetype.FT_PIXEL_MODE_MONO


def test_unsupported_render_mode_is_explicit():
    face = mft.Face(_font())
    face.set_pixel_sizes(0, 20)
    face.load_char("A", freetype.FT_LOAD_DEFAULT)
    with pytest.raises(NotImplementedError):
        face.glyph.render(freetype.FT_RENDER_MODE_LCD)


def test_non_outline_render_is_explicit():
    face = mft.Face(_font())
    face._face = SimpleNamespace(glyph=SimpleNamespace(format=0))
    with pytest.raises(NotImplementedError, match="scalable outline"):
        face._finish_load(freetype.FT_LOAD_RENDER, 1)


@pytest.mark.parametrize("char", ["A", "g"])
def test_light_target_parity(char):
    ours = mft.Face(_font())
    ours.set_pixel_sizes(0, 28)
    ours.load_char(
        char, freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_LIGHT
    )
    reference = freetype.Face(_font())
    reference.set_pixel_sizes(0, 28)
    reference.load_char(char, freetype.FT_LOAD_TARGET_LIGHT)
    reference.glyph.render(freetype.FT_RENDER_MODE_LIGHT)
    bitmap = reference.glyph.bitmap
    expected = np.asarray(bitmap.buffer, dtype=np.uint8).reshape(
        bitmap.rows, bitmap.pitch
    )[:, : bitmap.width]
    got = ours.glyph.bitmap.to_array()
    assert got.shape == expected.shape
    assert np.abs(got.astype(np.int16) - expected.astype(np.int16)).mean() < 3.0


def test_simd_tail_render_parity():
    vector_width = int(lib().mft_simd_width_float64())
    ours, left, top = _ours("A", 18)
    reference, ref_left, ref_top = _upstream("A", 18)
    assert ours.shape[1] % vector_width != 0
    assert (left, top, ours.shape) == (ref_left, ref_top, reference.shape)
    difference = np.abs(ours.astype(np.int16) - reference.astype(np.int16))
    assert difference.mean() < 3.0
    assert np.quantile(difference, 0.99) <= 24


@pytest.mark.parametrize("width", [16, 13])
def test_simd_mono_pack_full_vectors_and_tail(width):
    gray = np.arange(3 * width, dtype=np.uint8).reshape(3, width) * 17
    pitch = ((width + 15) // 16) * 2
    packed = np.empty((3, pitch), dtype=np.uint8)
    lib().mft_pack_mono(
        addr(gray), addr(packed), width, 3, width, pitch
    )
    expected = np.zeros_like(packed)
    bits = np.packbits(gray >= 128, axis=1, bitorder="big")
    expected[:, : bits.shape[1]] = bits
    assert np.array_equal(packed, expected)


@pytest.mark.parametrize(
    ("config", "error"),
    [
        (mft.RasterConfig(samples=0), ValueError),
        (mft.RasterConfig(samples=1.5), TypeError),
        (mft.RasterConfig(curve_tolerance=float("nan")), ValueError),
        (mft.RasterConfig(max_curve_depth=-1), ValueError),
        (mft.RasterConfig(parallel_threshold=1 << 63), OverflowError),
    ],
)
def test_invalid_config_is_rejected_before_ffi(config, error):
    face = freetype.Face(_font())
    face.set_pixel_sizes(0, 20)
    face.load_char("A", freetype.FT_LOAD_DEFAULT)
    with pytest.raises(error):
        mft.rasterize(face.glyph.outline, config=config)


def test_invalid_prepared_segment_layout_is_rejected():
    face = freetype.Face(_font())
    face.set_pixel_sizes(0, 20)
    face.load_char("A", freetype.FT_LOAD_DEFAULT)
    bad = np.empty((5, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="C-contiguous float64"):
        mft.rasterize(face.glyph.outline, _segments=bad)


def test_even_odd_fill_rule():
    class Outline:
        n_points = 8

        def __init__(self, flags):
            self.flags = flags

        def get_bbox(self):
            return SimpleNamespace(xMin=0, yMin=0, xMax=128, yMax=128)

        def decompose(self, move_to, line_to, **_callbacks):
            points = [
                SimpleNamespace(x=0, y=0),
                SimpleNamespace(x=128, y=0),
                SimpleNamespace(x=128, y=128),
                SimpleNamespace(x=0, y=128),
            ]
            for _ in range(2):
                move_to(points[0], None)
                for point in points[1:]:
                    line_to(point, None)

    winding, _, _ = mft.rasterize(Outline(0))
    even_odd, _, _ = mft.rasterize(
        Outline(freetype.FT_OUTLINE_EVEN_ODD_FILL)
    )
    assert winding.to_array()[0, 0] == 255
    assert even_odd.to_array()[0, 0] == 0


def test_parallel_threshold_matches_serial():
    face = freetype.Face(_font())
    face.set_pixel_sizes(0, 128)
    face.load_char("@", freetype.FT_LOAD_DEFAULT)
    serial, serial_left, serial_top = mft.rasterize(
        face.glyph.outline,
        config=mft.RasterConfig(parallel_threshold=10**18),
    )
    parallel, parallel_left, parallel_top = mft.rasterize(
        face.glyph.outline,
        config=mft.RasterConfig(parallel_threshold=0),
    )
    assert (parallel_left, parallel_top) == (serial_left, serial_top)
    assert np.array_equal(parallel.to_array(), serial.to_array())


def test_face_reuses_and_invalidates_prepared_segments(monkeypatch):
    calls = 0
    original = mft.flatten_outline

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(mft, "flatten_outline", counted)
    face = mft.Face(_font())
    face.set_pixel_sizes(0, 32)
    face.load_char("A")
    face.load_char("A")
    assert calls == 1
    face.set_pixel_sizes(0, 48)
    face.load_char("A")
    assert calls == 2


def test_face_reuses_raster_scratch_allocations():
    face = mft.Face(_font())
    face.set_pixel_sizes(0, 128)
    face.load_char("@")
    scratch_ids = (
        id(face._scratch.intersections),
        id(face._scratch.directions),
        id(face._scratch.active_edges),
        id(face._scratch.coverage),
    )
    face.load_char("@")
    assert scratch_ids == (
        id(face._scratch.intersections),
        id(face._scratch.directions),
        id(face._scratch.active_edges),
        id(face._scratch.coverage),
    )


@pytest.mark.parametrize("char", ["S", "g", "@", "8"])
@pytest.mark.parametrize("size", [12, 48])
def test_cff_cubic_outline_parity(char, size):
    path = Path(_matching_font("Nimbus Roman"))
    ours = mft.Face(str(path))
    ours.set_pixel_sizes(0, size)
    ours.load_char(char)
    reference = freetype.Face(str(path))
    reference.set_pixel_sizes(0, size)
    reference.load_char(char, freetype.FT_LOAD_DEFAULT)
    reference.glyph.render(freetype.FT_RENDER_MODE_NORMAL)
    ref_bitmap = reference.glyph.bitmap
    ref_array = np.asarray(ref_bitmap.buffer, dtype=np.uint8).reshape(
        ref_bitmap.rows, ref_bitmap.pitch
    )[:, : ref_bitmap.width]
    got = ours.glyph.bitmap.to_array()
    assert (
        ours.glyph.bitmap_left,
        ours.glyph.bitmap_top,
        got.shape,
    ) == (
        reference.glyph.bitmap_left,
        reference.glyph.bitmap_top,
        ref_array.shape,
    )
    difference = np.abs(got.astype(np.int16) - ref_array.astype(np.int16))
    assert difference.mean() < 5.0
    assert np.mean((got > 8) == (ref_array > 8)) > 0.96
