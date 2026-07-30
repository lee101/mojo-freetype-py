"""ctypes binding for the Mojo scan converter."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
LIB = os.environ.get("MOJOFREETYPE_LIB") or os.path.join(
    ROOT, "dist", "libmojo-freetype-py.so"
)

I = ctypes.c_int64

_SIGNATURES = {
    "mft_raster_gray": ([I] * 15, None),
    "mft_pack_mono": ([I] * 6, None),
    "mft_simd_width_float64": ([], I),
}


class BuildError(RuntimeError):
    pass


def _sources() -> list[str]:
    return [
        os.path.join(directory, name)
        for directory, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]


def build(force: bool = False) -> str:
    if os.environ.get("MOJOFREETYPE_LIB") and os.path.exists(LIB) and not force:
        return LIB
    sources = _sources()
    if not sources:
        if os.path.exists(LIB):
            return LIB
        raise BuildError(f"no Mojo sources found at {SRC}")
    if not force and os.path.exists(LIB):
        if os.path.getmtime(LIB) >= max(os.path.getmtime(path) for path in sources):
            return LIB
    pixi = shutil.which("pixi")
    if not pixi:
        raise BuildError("pixi not found; run `pixi run build` or set MOJOFREETYPE_LIB")
    proc = subprocess.run(
        [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "build"],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def addr(array: np.ndarray) -> int:
    if not isinstance(array, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if not array.flags.c_contiguous:
        raise ValueError("FFI buffers must be C-contiguous")
    if array.size == 0 or array.ctypes.data == 0:
        raise ValueError("FFI buffers must be non-empty and non-null")
    return int(array.ctypes.data)
