"""PhysMorph v2 — differentiable elastoplastic MPM morphing (Warp + torch).

See docs/SPEC.md for the full specification. Equation numbers (e.g. (5)) refer to it.
"""
from __future__ import annotations

import warp as wp

# Single global Warp init. Safe to call repeatedly (Warp guards re-init).
wp.init()

__all__ = ["wp"]
__version__ = "2.0.0-dev"
