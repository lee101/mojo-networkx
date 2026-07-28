from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.path.join(ROOT, "dist", "libmojo-networkx.so")
I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mnx_bfs": ([I] * 9, I),
    "mnx_dfs": ([I] * 12, I),
    "mnx_components": ([I] * 5, I),
    "mnx_scc": ([I] * 10, I),
    "mnx_dijkstra": ([I, I, I, I, I, F, I, I, I, I, I], I),
    "mnx_betweenness": ([I] * 10, None),
    "mnx_betweenness_weighted": ([I] * 12, None),
    "mnx_closeness_unweighted": ([I] * 7, None),
    "mnx_closeness_weighted": ([I] * 11, None),
    "mnx_pagerank": ([I, I, I, I, F, I, F, I, I, I], I),
    "mnx_eigenvector": ([I, I, I, I, I, F, I, I], I),
}

_handle: ctypes.CDLL | None = None


def build() -> str:
    proc = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        output = "\n".join(part.strip() for part in (proc.stdout, proc.stderr) if part.strip())
        raise RuntimeError(output or f"Mojo build failed with exit status {proc.returncode}")
    return LIB


def lib() -> ctypes.CDLL:
    global _handle
    if _handle is None:
        if not os.path.exists(LIB):
            build()
        _handle = ctypes.CDLL(LIB)
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_handle, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _handle


def addr(array: np.ndarray) -> int:
    if not isinstance(array, np.ndarray) or array.ndim != 1:
        raise TypeError("FFI buffers must be one-dimensional NumPy arrays")
    if array.dtype not in (np.dtype(np.int64), np.dtype(np.float64)):
        raise TypeError("FFI buffers must have native int64 or float64 dtype")
    if not array.dtype.isnative:
        raise TypeError("FFI buffers must use native byte order")
    if not array.flags.c_contiguous or not array.flags.aligned or not array.flags.writeable:
        raise ValueError("FFI buffers must be contiguous, aligned, and writable")
    address = int(array.ctypes.data)
    if address == 0:
        raise ValueError("FFI buffers must have a non-null data pointer")
    return address


def _empty(dtype: np.dtype, size: int) -> np.ndarray:
    if size < 0:
        raise ValueError("buffer size must be non-negative")
    # A zero-length allocation may legally expose address 0. Keep a one-element
    # owner and return an empty view so Mojo can always construct its non-null
    # UnsafePointer, even for null/edgeless graphs.
    array = np.empty(max(1, size), dtype=dtype)
    return array if size else array[:0]


def i64(size_or_values) -> np.ndarray:
    if isinstance(size_or_values, int):
        return _empty(np.dtype(np.int64), size_or_values)
    return np.ascontiguousarray(size_or_values, dtype=np.int64)


def f64(size_or_values) -> np.ndarray:
    if isinstance(size_or_values, int):
        return _empty(np.dtype(np.float64), size_or_values)
    return np.ascontiguousarray(size_or_values, dtype=np.float64)
