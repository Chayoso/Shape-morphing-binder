"""Owned diagnostic snapshots of the paced density and CIC/PBR observations.

This is not an alternative optimization objective. It permits a reference swap
on one observed path without rebuilding its plan, neighbors, or observations.
"""
from dataclasses import dataclass

import torch

from ..losses.volumetric import d_vol_density
from .render_loss import d_render, d_pbr


def own(value, device):
    return torch.as_tensor(value, device=device).detach().clone()


@dataclass(frozen=True)
class PreparedReference:
    density: dict
    render: dict
    shade: dict | None
    pbr_weight: float
    kind: str

    @classmethod
    def capture(cls, cfg, tgt, grid, alphas, shades, pbr_grid, kind):
        if cfg.loss_units != 'density' or cfg.phys_loss != 'ot_pace':
            raise ValueError('Reference audit requires density-unit ot_pace')
        if tgt.gauss is not None or tgt.surface_gs is not None or alphas is None:
            raise ValueError('Reference audit supports active CIC/PBR observations only')
        dev = grid.device
        density = dict(m=own(tgt.m, dev), target_grid=own(grid, dev),
                       grid_min=own(tgt.lgmin, dev), dx=float(tgt.ldx),
                       dims=tuple(tgt.ldims), m_ref=float(tgt.m_ref),
                       n_support=int(tgt.n_support), form=cfg.dvol_form)
        render = dict(target_alphas=tuple(own(a, dev) for a in alphas),
                      views=tuple(tuple(v) for v in tgt.views), res=cfg.render_res,
                      extent=float(tgt.extent), k=cfg.sil_k,
                      w_hole=cfg.w_hole, w_spray=cfg.w_spray)
        shade = None
        if cfg.w_pbr > 0 and shades is not None:
            fine = cfg.pbr_denoised and bool(tgt.pdims) and pbr_grid
            shade = dict(shade_tgts=tuple(own(s, dev) for s in shades),
                         views=render['views'], res=render['res'], extent=render['extent'],
                         grid_min=own(tgt.pgmin if fine else tgt.lgmin, dev),
                         dx=float(tgt.pdx if fine else tgt.ldx),
                         dims=tuple(tgt.pdims if fine else tgt.ldims), k=cfg.sil_k,
                         ambient=cfg.pbr_ambient, blur_cells=float(tgt.pblur) if fine else 0.)
        return cls(density, render, shade, float(cfg.w_pbr), kind)

    def terms(self, x):
        volume = d_vol_density(x, **self.density)
        silhouette = d_render(x, **self.render)
        shade = x.sum()*0. if self.shade is None else d_pbr(x, **self.shade)
        return dict(volume=volume, silhouette=silhouette, pbr=shade,
                    render=silhouette+self.pbr_weight*shade)
