"""Benchmark Mojo scan conversion against upstream FreeType."""

from __future__ import annotations

import math
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"
    ),
)

import freetype  # noqa: E402
import mojofreetype as mft  # noqa: E402


TEXT = "Mojo rasterization: AVWgy@éЖ0123456789"


def benchmark_font() -> Path:
    result = subprocess.run(
        ["fc-match", "-f", "%{file}", "DejaVu Sans"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip())


def timeit(function, repeat: int = 5) -> float:
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def renderer(module, size: int, count: int, mono: bool = False):
    font = benchmark_font()
    face = module.Face(str(font))
    face.set_pixel_sizes(0, size)
    flags = freetype.FT_LOAD_RENDER
    if mono:
        flags |= freetype.FT_LOAD_TARGET_MONO | freetype.FT_LOAD_MONOCHROME
    chars = (TEXT * (count // len(TEXT) + 1))[:count]

    def render():
        checksum = 0
        for char in chars:
            face.load_char(char, flags)
            checksum += face.glyph.bitmap.width
        return checksum

    return render


def main() -> None:
    font = benchmark_font()
    if not font.is_file():
        raise SystemExit("benchmark font (DejaVu Sans) not found")
    cases = [
        ("grayscale 16 px, 200 glyphs", 16, 200, False),
        ("grayscale 48 px, 100 glyphs", 48, 100, False),
        ("grayscale 128 px, 40 glyphs", 128, 40, False),
        ("mono 48 px, 100 glyphs", 48, 100, True),
    ]
    print("| case | Mojo | FreeType | ratio |")
    print("| --- | ---: | ---: | ---: |")
    for name, size, count, mono in cases:
        ours = renderer(mft, size, count, mono)
        reference = renderer(freetype, size, count, mono)
        ours()
        reference()
        mojo_seconds = timeit(ours)
        reference_seconds = timeit(reference)
        ratio = reference_seconds / mojo_seconds
        result = "faster" if ratio >= 1.0 else "slower"
        print(
            f"| {name} | {mojo_seconds * 1e3:.2f} ms | "
            f"{reference_seconds * 1e3:.2f} ms | "
            f"{ratio:.3f}x ({result}) |"
        )


if __name__ == "__main__":
    main()
