"""PhysMorph — render-guided differentiable MPM shape morphing (Warp + torch, CUDA only).

README.md describes the pipeline; physmorph.gpu the device helpers of the run stage.
"""
from __future__ import annotations

import sys as _sys

if "torch" not in _sys.modules:
    # CuPy compiles its device kernels with NVRTC. The server's torch environment also carries
    # a CUDA 11 NVRTC that torch loads globally; loaded first, CuPy would bind to it and fail on
    # its CUDA 12 headers. Load CuPy's CUDA 12 NVRTC before torch (physmorph.gpu checks it).
    try:
        import cupy as _cupy  # noqa: F401  (sets up CuPy's CUDA library search)
        from cupy_backends.cuda.libs import nvrtc as _nvrtc
        _nvrtc.getVersion()
    except ImportError:  # no CuPy (the render-only tools): nothing to order
        pass

import warp as wp  # noqa: E402

# Single global Warp init. Safe to call repeatedly (Warp guards re-init).
wp.init()

__all__ = ["wp"]
__version__ = "3.0.0"
