"""Fixed-target render merit for the positions proposed for an outer commit.

Inner losses may use a moving paced target or include shading/Gaussian terms.
They identify channel eligibility here; their numerical values are not merit.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .render_loss import d_render

if TYPE_CHECKING:
    from .config import PipelineConfig
    from .optimizer import TargetPack


@torch.no_grad()
def fixed_outer_render(
    x: torch.Tensor,
    cfg: PipelineConfig,
    tgt: TargetPack,
    inner_sil: float | torch.Tensor | None,
    inner_render: float | torch.Tensor | None,
) -> torch.Tensor | None:
    """Pure silhouette merit on promoted ``x``, retaining its Torch device.

    Both inner channels being None means rendering is inactive for this window,
    even if a target pack remains cached. A zero-valued channel is still active.
    The caller owns acceptance/history and any scalar telemetry conversion.
    """
    if inner_sil is None and inner_render is None:
        return None
    if tgt.sils is None or len(tgt.sils) == 0:
        raise ValueError("active outer render merit requires fixed tgt.sils")
    if len(tgt.sils) != len(tgt.views):
        raise ValueError("fixed tgt.sils must contain one image per target view")
    for alpha in tgt.sils:
        if not isinstance(alpha, torch.Tensor):
            raise TypeError("fixed tgt.sils must be Torch tensors; CPU array conversion is not allowed")
        if alpha.device != x.device:
            raise ValueError("fixed tgt.sils and promoted positions must share a device")
        if alpha.shape != (cfg.render_res, cfg.render_res):
            raise ValueError("fixed tgt.sils resolution must match cfg.render_res")
    return d_render(x, tgt.sils, tgt.views, cfg.render_res, tgt.extent,
                    cfg.sil_k, cfg.w_hole, cfg.w_spray)
