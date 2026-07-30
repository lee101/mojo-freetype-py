# mojo-freetype-py

`mojo-freetype-py` is a standalone Mojo implementation of glyph outline scan
conversion with a Python API matching the rasterization path in
[`freetype-py`](https://github.com/rougier/freetype-py).

The package is imported as `mojofreetype`. Its covered `Face`, `GlyphSlot`, and
`Bitmap` surface normally needs only an import change:

```python
import subprocess

import mojofreetype as freetype

font_path = subprocess.run(
    ["fc-match", "-f", "%{file}", "DejaVu Sans"],
    check=True,
    capture_output=True,
    text=True,
).stdout
face = freetype.Face(font_path)
face.set_pixel_sizes(0, 48)
face.load_char("A")

bitmap = face.glyph.bitmap
coverage = bitmap.to_array()
print(bitmap.width, bitmap.rows, coverage.max())
```

`to_array()` is a convenience extension. Upstream-compatible `rows`, `width`,
`pitch`, `buffer`, `num_grays`, `pixel_mode`, `palette_mode`, and `palette`
properties are also available.

## Coverage

| API | Status |
| --- | --- |
| `Face(path_or_stream, index=0)` | Delegates font parsing and face management to `freetype-py` |
| `Face.load_char(char, flags=FT_LOAD_RENDER)` | Mojo rendering for scalable outlines |
| `Face.load_glyph(index, flags=FT_LOAD_RENDER)` | Mojo rendering for scalable outlines |
| `Face.glyph` and glyph metrics | Compatible proxy over the upstream glyph slot |
| `GlyphSlot.render(render_mode)` | Normal, light, and monochrome |
| `Bitmap` | Gray and mono layout and metadata |
| `rasterize(outline, render_mode=..., origin=None)` | Direct outline-to-bitmap API |
| TrueType quadratic and CFF cubic outlines | Adaptive flattening and non-zero winding fill |
| `FT_OUTLINE_EVEN_ODD_FILL` | Even-odd fill |

FreeType still handles font parsing, character maps, scaling, transforms,
hinting, and glyph metrics. This repository replaces the scan converter, not
the font driver. LCD/LCD-V subpixel rendering, SDF rendering, color-layer
composition, stroking, SVG glyphs, and embedded bitmap strikes are not
covered. Unsupported render modes and non-outline glyphs fail explicitly.

## Install

The supported installation is a source checkout on Linux x86-64:

```bash
git clone https://github.com/lee101/mojo-freetype-py
cd mojo-freetype-py
pixi install
pixi run build
```

Pixi installs the pinned Mojo nightly, Python, NumPy, and `freetype-py`. The
build task emits `dist/libmojo-freetype-py.so`. Set `MOJOFREETYPE_LIB` to use a
prebuilt library. Wheels and macOS/Windows builds are not currently provided.

The test suite renders the same hinted outlines through this package and
FreeType. It covers quadratic TrueType and cubic CFF glyphs, winding and
even-odd fill, accents, multiple writing systems, empty glyphs, normal, light,
and mono rendering, loading by character and glyph index, stream-backed faces,
metrics proxying, cache invalidation, SIMD tails, and the bitmap property
contract. Tests assert identical bitmap bounds plus numerical coverage error
and ink-mask overlap.

```bash
pixi run build && pixi run test
```

## Benchmarks

Measured by `pixi run bench` on this checkout's publication machine. Each row
is the best of five complete glyph batches after warming both renderers.

| case | Mojo | FreeType | ratio |
| --- | ---: | ---: | ---: |
| grayscale 16 px, 200 glyphs | 16.41 ms | 2.48 ms | 0.151x (slower) |
| grayscale 48 px, 100 glyphs | 8.82 ms | 1.03 ms | 0.116x (slower) |
| grayscale 128 px, 40 glyphs | 10.51 ms | 1.21 ms | 0.115x (slower) |
| mono 48 px, 100 glyphs | 16.33 ms | 1.96 ms | 0.120x (slower) |

FreeType remains faster. Prepared edges are cached per face and invalidated
when face state changes. The scan converter builds a SIMD-filtered active-edge
list once per row, vectorizes coverage and mono packing with scalar tails, and
can use independent row workers for sufficiently large glyphs. `pixi run
bench` serializes benchmark runs with a machine-wide lock. There is no GPU
path.

## How it works

FreeType loads, scales, hints, and decomposes the selected glyph outline.
Quadratic and cubic Bézier curves are adaptively flattened into a contiguous
row-major NumPy array of line-segment endpoints. Non-horizontal edges are
prepared into a structure-of-arrays layout with bounds, inverse slopes, and
winding directions, then cached by `Face`.

Python makes one synchronous `ctypes` call into the Mojo shared library.
Python validates and owns every input, output, and scratch allocation for the
duration of that call, so NumPy storage crosses the FFI boundary without a
copy.

For each output row, Mojo filters the prepared edges, intersects and sorts the
active candidates for each sample, applies the outline's winding rule, and
accumulates horizontal pixel coverage. Rows are stored top-to-bottom as
contiguous gray values. Mono mode thresholds that coverage and packs bits in
FreeType-compatible order and pitch.

## License

MIT
