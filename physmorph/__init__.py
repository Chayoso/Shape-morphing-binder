"""PhysMorph v2 — differentiable elastoplastic MPM morphing (Warp + torch).

See docs/SPEC.md for the full specification. Equation numbers (e.g. (5)) refer to it.
"""
from __future__ import annotations

import os
import warp as wp

# Older code generators emit a nested custom adjoint after its first caller.
# The constitutive rotation adjoint requires the ordering supported in 1.16.
if tuple(int(part) for part in wp.__version__.split('.')[:2]) < (1, 16):
    raise RuntimeError('PhysMorph requires warp-lang>=1.16 for the constitutive custom adjoint')

# Warp does not consume WARP_CACHE_PATH itself. Honor the launcher's /data path
# before initialization can create its default cache outside the run workspace.
if os.environ.get("WARP_CACHE_PATH"):
    wp.config.kernel_cache_dir = os.environ["WARP_CACHE_PATH"]

# Single global Warp init. Safe to call repeatedly (Warp guards re-init).
wp.init()

__all__ = ["wp"]
__version__ = "2.0.0-dev"
