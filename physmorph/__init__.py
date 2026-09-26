"""PhysMorph v2 — differentiable elastoplastic MPM morphing (Warp + torch).

See docs/SPEC.md for the full specification. Equation numbers (e.g. (5)) refer to it.
"""
from __future__ import annotations

import os
import warp as wp

# Warp does not consume WARP_CACHE_PATH itself. Honor the launcher's /data path
# before initialization can create its default cache outside the run workspace.
if os.environ.get("WARP_CACHE_PATH"):
    wp.config.kernel_cache_dir = os.environ["WARP_CACHE_PATH"]

# Single global Warp init. Safe to call repeatedly (Warp guards re-init).
wp.init()

__all__ = ["wp"]
__version__ = "2.0.0-dev"
