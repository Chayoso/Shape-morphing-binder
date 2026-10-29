"""The fixed target of a run: grids, images and calibrations built once on the device.

TargetPack holds what every window reads: the target mass on the loss grid, the render
targets (silhouettes and the matched shading images), the target's distance transform for
the W1 cleanup, the target points and their device KD-tree for the near-band cleanup, the
surface-proximity term, and the scalars measured once at the source (the unit ratios, the
transport scale, the render weight).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .. import gpu
from ..losses.silhouette import set_kernel
from ..losses.support import SurfaceProximity
from ..losses.volumetric import (d_vol, d_vol_density, density_units, rasterize_mass_cubic, target_dt_grid,
                                 target_mass_grid)
from ..mpm.state import MPMParams
from .config import PipelineConfig
from ..render.exterior import Lattice, found_disc_albedo, make_field
from .render_loss import exterior_targets, make_views, shade_targets, target_silhouettes


@dataclass
class Exterior:
    """What the render terms need when they are read on the exterior (render/exterior.py): the field's pitch, the
    lattice and its pitch, the skin of the discs' particle lists, and the target's images from its own discs."""
    pitch: float
    lattice: Lattice
    h: float
    skin: float
    sils: list
    shade: list
    radius: float = 3.0                 # the field's kernel radius and offset, in pitches (D123; 3 and 0.8 = D59's)
    offset: float = 0.8
    field: str = "zb"                   # D140: "zb" Zhu and Bridson's field, "aniso" the anisotropic one
    albedo: torch.Tensor | None = None  # D160 (appearance): the body particles' fixed albedos (N, C), carried by index
    albedo_ref: torch.Tensor | None = None  #   and their mean (C,), about which the discs' albedos are written
    colour: str = "kernel"              # D173: the discs' colour weights, "kernel" (D160's, the frozen path) or "nearest"
    colour_tau: float | None = None     #   (render/exterior.nearest_weights, tau = half the exterior's pitch)
    counts: list | None = None          # D175b (keyframe_compare column, a keyframe's exterior only): its projected count
                                        #   pictures; the render term is then render_loss.d_count on the particles
    mask_sil: torch.Tensor | None = None    # D178 (render_split "masked"): the content mask of the silhouettes (V,R,R) and
    mask_shade: torch.Tensor | None = None  #   of the shading ((V,R,R), or (C,V,R,R) with colour); None = no content there
    mask_info: dict | None = None           #   the floors and the masks' cover (a record)


@dataclass
class TargetPack:
    grid: torch.Tensor                  # target mass on the loss grid
    lgmin: torch.Tensor
    ldx: float
    ldims: tuple
    m: torch.Tensor                     # unit loss-side masses (N,)
    views: list                         # [(theta, phi)]
    sils: list                          # target alpha images
    extent: float                       # render half-extent (1.25 x max |target|)
    shade: list                         # matched shading images
    pgmin: torch.Tensor                 # the morph's normal grid at the render pixel,
    pdx: float                          #   blurred by pblur cells (the drawn surface)
    pdims: tuple
    pblur: float
    dt3: torch.Tensor                   # target DT on its own fine grid (W1 cleanup)
    dtgmin: torch.Tensor
    dtdx: float
    dtdims: tuple
    pts: torch.Tensor                   # target points (N,3)
    knn: gpu.KNN                        # device KD-tree of pts
    nn_spacing: float                   # target median nearest-neighbour spacing
    m_ref: float                        # density-unit constants of the loss grid
    n_support: int
    support: SurfaceProximity
    unit_ratio: float = 1.0             # D_vol(legacy) / D_vol(density) at the source
    unit_grad_ratio: float = 1.0        # |grad D_vol(legacy)| / |grad D_vol(density)|
    grid_ot: object = None              # GridSinkhornLoss, built at the first window
    ot_scale: float | None = None       # transport scale: equal gradient norm with D_vol
    settled_step: float | None = None   # last accepted step (warm start of the search)
    u_adam: tuple | None = None         # D146b (u_adam_carry): u's Adam state (m, v, age) after the last committed window
    force_ratio: float | None = None    # B13 (baseline force*): |grad_f phys| / |grad_dFc phys| at the source (the first
                                        #   gradient), the step control's constants in the force leaf's units
    gate: tuple | None = None           # (grid, dx, dims) of the u transport gate: the MPM-cell grid
    ext: Exterior | None = None         # the render terms' exterior, when they are read on it
    relief: object = None               # window/layer.TargetRelief: the relaxation's reference, from the target's surface
    xu: tuple | None = None             # cfg.baseline "xu": (the target mass on the simulation grid, its origin, dx, dims)
    draws: list | None = None           # further samples of the target whose pictures the render's targets average
    gap: tuple | None = None            # D148 (gap_split): (plane_m, plane_n) per MPM node of mpm/gap_split.GapPlanes (torch)
    gap_walls: dict | None = None       # D148c (gap_split walls): gap_walls.BoxPieces' arrays (torch)
    gap_rep: object = None              # D148c: each particle's representative cell from the window before (state)
    render_floor: float | None = None   # D162 (1): the render term of the pictures' own first sample (render_floor)
    render_b: float | None = None       # D162b: the shape reading b at the current window's start (shape_read.ShapeRead)
    render_b_se: float | None = None    #   its standard error over the eight resamples
    render_b_floor: float | None = None  #   the floor sample read the same way, once a run
    render_lp: float | None = None      # D162d: the pitch-low-passed render term at the current window's start
    render_lp_floor: float | None = None  #   the floor sample read the same way, once a run
    geometry: bool = True               # D175: the final target's geometry (transport, proximity) and cleanup are read;
                                        #   False in a keyframe's segment (only the released motion and the end drift)
    keyframe: int | None = None         # D175: the keyframe whose segment the window serves (None: the final target)
    soundness: bool = True              # D175c: the released motion and the end drift are read (False: a segment owned by
                                        #   its keyframe alone, keyframe_soundness "final")


def target_relief(tgt_t: torch.Tensor, surface, cfg: PipelineConfig):
    """The relaxation's reference (window/layer.TargetRelief) from points of the target mesh's surface and their
    normals. The mesh's normals are taken as they are, all turned at once if as a whole they point into the
    sample (point by point the sample cannot tell: its eight nearest particles put 12-15 % of a sound mesh's
    normals on the wrong side, and turning those gave residuals of tens of pitches, D88). Printed: the reference
    and the rough residual d - dbar on the target sample's own layer, which should share their mean (D105)."""
    from .window.layer import TargetRelief, layer_relax_data, layer_spacing
    pts, nrm = (gpu.tensor(np.asarray(v, np.float32)) for v in surface)
    inside = tgt_t[gpu.KNN(tgt_t).query(pts, 8)[1]].mean(1)
    if float(((pts - inside) * nrm).sum(1).sign().mean()) < 0:
        nrm = -nrm
    sp = layer_spacing(tgt_t)
    relief = TargetRelief(pts, nrm, sp)
    mask, lnrm, nbr, w = layer_relax_data(tgt_t, sp, k=cfg.layer_k, h_sp=cfg.layer_h_sp)
    on = mask > 0.5
    d = (lnrm * (tgt_t - (w[..., None] * tgt_t[nbr]).sum(1))).sum(1)
    rough, ref = (d - (w * d[nbr]).sum(1))[on] / sp, relief.at(tgt_t, mask, lnrm, nbr, w)[on] / sp
    print(f"[target] relief: {len(pts)} surface points; on the target sample's layer (spacings) the reference has mean "
          f"{float(ref.mean()):+.4f}, rms {float(ref.square().mean().sqrt()):.4f}, the rough residual mean "
          f"{float(rough.mean()):+.4f}, rms {float(rough.square().mean().sqrt()):.4f}", flush=True)
    return relief


def shell_pictures(mesh, lattice, h, pitch, radius, refs, views, extent, cfg, sils0, shade0):
    """D159: the render's target pictures drawn from the target mesh grown by the run's own exterior offset (the signed
    distance from the mesh of refs, the run's target exterior discs, averaged over the field's radius `radius`, wu), on
    the exterior's lattice by the same operators; sils0, shade0 (the samples' pictures) only for the record."""
    if mesh is None:
        raise ValueError("render_target mesh_shell needs the target mesh in the target's frame")
    from ..render.mesh_shell import mesh_shell_discs
    p, n, c, info = mesh_shell_discs(lattice, h, mesh[0], mesh[1], pitch, refs, radius)
    sils, shade = exterior_targets(p, n, views, cfg.render_res, extent, cfg.sil_k, cfg.pbr_ambient)
    msq = lambda a, b: float(torch.stack([(x - y).square().mean() for x, y in zip(a, b)]).mean())   # noqa: E731
    print(f"[target] render target (D159): the target mesh grown by the run's own exterior offset, local over the field's "
          f"radius: c mean {c / pitch:.3f} pitches (sd {info['c_sd'] / pitch:.3f}, p01 {info['c_p01'] / pitch:.3f}, median "
          f"{info['c_p50'] / pitch:.3f}, p99 {info['c_p99'] / pitch:.3f}) from {sum(len(r) for r in refs)} discs of "
          f"{len(refs)} samples in {info['cells']} cells; the mesh {info['faces']} faces, {info['loops_closed']} loops "
          f"closed, watertight {info['watertight']}; {info['discs']} discs; its pictures against the samples' (mean "
          f"square): silhouette {msq(sils, sils0):.3e}, shading {msq(shade, shade0):.3e}", flush=True)
    return sils, shade


def sample_pictures(s, cfg, pitch, radius, offset, lattice, h, views, extent, alb=None, b_ref=None):
    """One sample's exterior pictures (build_target's operators): the field at the pitch, its discs on the lattice, and
    the silhouettes and shading drawn from them (albedo: the discs' colours, D160). Returns (discs, their colours or
    None, (silhouettes, shading))."""
    with torch.no_grad():
        fld = make_field(cfg.exterior_field, s, pitch, radius=radius, offset=offset,
                         device_field=cfg.exterior_field == "zbs")
        p_t, g_t, _, _ = lattice.discs(fld, h, refine=False)
        c_t = None if alb is None else found_disc_albedo(fld, p_t, alb, b_ref, pitch, getattr(cfg, "colour_kernel", "kernel"),
                                                         0.5 * pitch)
    return p_t, c_t, exterior_targets(p_t, torch.nn.functional.normalize(g_t, dim=1), views, cfg.render_res,
                                      extent, cfg.sil_k, cfg.pbr_ambient, albedo=c_t)


def keyframe_pictures(final: TargetPack, samples, cfg: PipelineConfig, views=None):
    """D175: a keyframe's pictures, the mean over its samples of what the final pictures' exterior operators draw (the
    final pack's pitch, kernel, lattice and camera box; views default to the run's). Returns (silhouettes, shading)."""
    e = final.ext
    per = [sample_pictures(gpu.tensor(np.asarray(s, np.float32)), cfg, e.pitch, e.radius, e.offset, e.lattice, e.h,
                           final.views if views is None else views, final.extent)[2] for s in samples]
    return ([torch.stack(v).mean(0) for v in zip(*[p[0] for p in per])],
            [torch.stack(v).mean(0) for v in zip(*[p[1] for p in per])])


def content_masks(ext: Exterior, content, cfg: PipelineConfig, views, extent: float):
    """D178 (render_split "masked", the user's rule of 2026-10-29): where the run's pictures T say more than the 3D target.
    content = (own, indep): the 3D target's own samples (its sample and its draws: the pictures the run would have without a
    render target or a paint, T0) and as many further independent draws (T0'). T0 and T0' are grey pictures drawn with the
    run's exterior constants (pitch, kernel, lattice, camera box); the floor of each term is max |T0 - T0'| over the pixels
    (what two independent samples of the 3D target differ by, D176: 0.45-0.59 silhouette, 0.20-0.32 shading), and the masks
    are m = 1[|T - T0| > floor] per pixel and term (with colour, per channel against the grey T0). Sets ext.mask_sil /
    ext.mask_shade (None where no pixel is set: the rule is then Hu's there) and ext.mask_info (a record)."""
    own, indep = content
    if ext.counts is not None:                # D178 (render_compare column): the content mask of the count pictures
        from .render_loss import count_pictures

        def cnt(samples):
            return torch.stack([count_pictures(gpu.tensor(np.asarray(s, np.float32)), views, cfg.render_res, extent)
                                for s in samples]).mean(0)
        with torch.no_grad():
            C0, C1, C = cnt(own), cnt(indep), torch.stack(list(ext.counts), 0)
            f_c = float((C0 - C1).abs().max())
            m_c = (C - C0).abs() > f_c
            body = float(((C0 > 0) | (C > 0)).sum())
        ext.mask_sil = m_c.float() if bool(m_c.any()) else None     # the count term's mask (render_terms' counts path)
        ext.mask_info = dict(floor_count=f_c, cover_count=float(m_c.sum()) / max(body, 1.0), own=len(own), indep=len(indep))
        print(f"[target] content mask (D178, counts): floor {f_c:.4f} (max |C0 - C0'|); the count pictures differ from "
              f"the 3D target's beyond it on {100 * ext.mask_info['cover_count']:.2f} % of the body's pixels", flush=True)
        return

    def grey(samples):
        per = [sample_pictures(gpu.tensor(np.asarray(s, np.float32)), cfg, ext.pitch, ext.radius, ext.offset,
                               ext.lattice, ext.h, views, extent)[2] for s in samples]
        return (torch.stack([torch.stack(list(v)).mean(0) for v in zip(*[p[0] for p in per])]),
                torch.stack([torch.stack(list(v)).mean(0) for v in zip(*[p[1] for p in per])]))
    with torch.no_grad():
        A0, S0 = grey(own)
        A1, S1 = grey(indep)
        f_sil, f_shade = float((A0 - A1).abs().max()), float((S0 - S1).abs().max())
        A = torch.stack(list(ext.sils), 0)
        S = torch.stack(list(ext.shade), 0)
        colour = ext.albedo is not None
        if colour:                                        # (V,C,R,R) -> (C,V,R,R), as d_exterior compares them
            S = S.movedim(1, 0).contiguous()
            m_s = (S - S0[None]).abs() > f_shade
        else:
            m_s = (S - S0).abs() > f_shade
        m_a = (A - A0).abs() > f_sil
        body = float(((A0 > 1e-2) | (A > 1e-2)).sum())
        cov_a = float(m_a.sum()) / max(body, 1.0)
        cov_s = float(m_s.float().mean(0).sum() if colour else m_s.sum()) / max(body, 1.0)
        ext.mask_sil = m_a.float() if bool(m_a.any()) else None
        ext.mask_shade = m_s.float() if bool(m_s.any()) else None
    ext.mask_info = dict(floor_sil=f_sil, floor_shade=f_shade, cover_sil=cov_a, cover_shade=cov_s, own=len(own),
                         indep=len(indep))
    print(f"[target] content mask (D178): floors silhouette {f_sil:.4f}, shading {f_shade:.4f} (max |T0 - T0'| over "
          f"{len(own)} / {len(indep)} samples of the 3D target); the pictures differ from the 3D target's beyond them on "
          f"{100 * cov_a:.2f} % (silhouette) / {100 * cov_s:.2f} % (shading) of the body's pixels", flush=True)


def keyframe_counts(final: TargetPack, samples, cfg: PipelineConfig, views=None) -> list:
    """D175b: a keyframe's projected count pictures (render_loss.count_pictures), the mean over its samples, in the final
    pack's views and camera box."""
    from .render_loss import count_pictures
    with torch.no_grad():
        c = torch.stack([count_pictures(gpu.tensor(np.asarray(s, np.float32)), final.views if views is None else views,
                                        cfg.render_res, final.extent) for s in samples]).mean(0)
    return list(c)


def build_target(target_x, prm: MPMParams, cfg: PipelineConfig, draws=None, mesh=None, appearance=None,
                 render_x=None, content=None) -> TargetPack:
    """draws: further independent samples of the target in its frame. The render term's target pictures are then the
    mean of the pictures the operator draws of every sample: what it draws of the mesh in expectation. Drawn from
    the one sample, a render arm fitted that sample's noise as well, and half its measured lead over the
    physics-only twin was that fit (D90, D91).
    mesh (D159, cfg.render_target "mesh_shell"): (V, F) of the target mesh in the target's frame; the exterior's target
    pictures are then drawn from the mesh grown by the run's own measured exterior offset (render/mesh_shell.py), on
    the same lattice; everything else is as on the frozen path.
    appearance (D160, pipeline/appearance.Appearance): the body's fixed albedos and the target samples' (the target's
    paint); the exterior's shading pictures are then colour pictures drawn from the target discs' albedos (the same
    kernel weights as the body's discs), and the Exterior carries the body's albedos. Needs render_exterior.
    render_x (D156; None = the frozen path): the render's own target sample in the target's frame (prepare's
    render_tgt); the pictures, and the constants they are drawn with (the orthographic box, the pitch, the
    exterior's lattice), are then those of render_x and the draws (its further draws), and everything else (the
    transport, the u gate, the cleanups, the surface proximity, the W1 cleanup's grid on the target's own box) is
    target_x's. With render_x = target_x every field is the frozen path's."""
    if appearance is not None and not cfg.render_exterior:
        raise ValueError("appearance (D160) colours the exterior's discs: it needs render_exterior")
    set_kernel("cic")
    tgt_t = gpu.tensor(target_x)
    rnd_t = tgt_t if render_x is None else gpu.tensor(render_x)       # the first sample of the render's pictures
    samples = [rnd_t] + [gpu.tensor(d) for d in (draws or [])]
    mean = lambda per: [torch.stack(v).mean(0) for v in zip(*per)]   # noqa: E731  (per sample, a list over views)
    N = tgt_t.shape[0]
    m = m_t = torch.ones(N, device=gpu.DEVICE)                # the unit loss-side masses: the body's (tgt.m) and the target's
    # the loss grid covers the MPM domain (scalar geometry, float32 like the grid itself)
    dmin = np.asarray(prm.grid_min, np.float32)
    dmax = dmin + prm.dx * np.array([prm.nx, prm.ny, prm.nz], np.float32)
    ldx = float((dmax - dmin).max() / cfg.loss_res)
    # the fine geometry read from the target surface (D161b: covered within the berth; D162 (2): within one loss cell)
    support = SurfaceProximity(tgt_t, berth_k=cfg.nn_berth_k if cfg.proximity_berth else None,
                               free_radius=ldx if cfg.placement_at_cell else None)
    ldims = (cfg.loss_res,) * 3
    lgmin = torch.tensor(dmin, device=gpu.DEVICE)
    grid = target_mass_grid(tgt_t, m_t, lgmin, ldx, ldims)
    # the u transport gate measures its radius in MPM cells, so its transport map is solved on the
    # MPM-cell grid; that is the loss grid itself unless the loss grid refines with N
    gate = (grid, ldx, ldims)
    if cfg.cell_shape > 0:
        # D137: the MPM cell follows N; the gate keeps the shape's cell (its grid and, in the objective, its radius)
        gn = int(round(float((dmax - dmin).max()) / cfg.cell_shape))
        gdx, gdims = float((dmax - dmin).max() / gn), (gn,) * 3
        gate = (target_mass_grid(tgt_t, m_t, lgmin, gdx, gdims), gdx, gdims)
    elif cfg.loss_follows_n and cfg.loss_res != prm.nx:
        gdx, gdims = float((dmax - dmin).max() / prm.nx), (prm.nx,) * 3
        gate = (target_mass_grid(tgt_t, m_t, lgmin, gdx, gdims), gdx, gdims)
    views = make_views(cfg.render_views, cfg.render_elevs)
    geom = rnd_t
    extent = float(geom.abs().max()) * 1.25
    sils = mean([target_silhouettes(s, views, cfg.render_res, extent, cfg.sil_k) for s in samples])
    # shading: the morph's normals on a render-pixel grid blurred by the renderer's 1.5 target
    # spacings; the target image is drawn by the same operator at the target (matched), so
    # the loss compares the operator with itself and no target/operator bias enters
    sp_t = gpu.median_kth_spacing(rnd_t, 8, subsample=20000)
    pdx = 2.0 * extent / cfg.render_res
    pdims = tuple(int(np.ceil((dmax - dmin).max() / pdx)) for _ in range(3))
    pblur = 1.5 * sp_t / pdx
    shade = mean([shade_targets(s, views, cfg.render_res, extent, lgmin, pdx, pdims,
                                cfg.sil_k, cfg.pbr_ambient, blur_cells=pblur) for s in samples])
    print(f"[target] shading target: spacing {sp_t:.4f}, normal grid {pdims[0]}^3 at {pdx:.4f} wu "
          f"({pdx / sp_t:.2f} spacings), blur {pblur:.2f} cells", flush=True)
    ext = None
    if cfg.render_exterior:
        # the field's pitch is the volume sample's (0.708 of the 8th-neighbour distance); the discs sit on a lattice
        # of half a render pixel, no coarser than the field resolves (its smallest body, a sphere of 0.8 pitches,
        # holds a node of any lattice up to 0.92 pitches); the discs' particle lists reach one pitch past the kernel.
        # D123 (--exterior_radius): the kernel radius in pitches, the offset in proportion (0.8 at 3), the lattice cap with
        # the offset; at 3 every number is the old one bit for bit
        pitch = 0.708 * sp_t
        radius = cfg.exterior_radius
        offset = 0.8 * (radius / 3.0)
        center = geom.mean(0)
        lattice = Lattice(center, 2.8 * float((geom - center).norm(dim=1).max()))
        h = min(extent / cfg.render_res, 0.92 * pitch * (radius / 3.0))
        per, refs = [], []
        b_alb = b_ref = None
        if appearance is not None:            # D160: the body's albedos, their mean, and each target sample's
            b_alb = gpu.tensor(np.asarray(appearance.src, np.float32))
            b_ref = b_alb.mean(0)
            alb = list(appearance.target_albedos())
            if render_x is not None:          # D172: the pictures' first sample is render_x (above): its paint, not target_x's
                alb[0] = appearance.albedo_at(np.asarray(render_x))
            t_alb = [gpu.tensor(np.asarray(a, np.float32)) for a in alb]
            if len(t_alb) != len(samples) or b_alb.shape != (N, t_alb[0].shape[1]):
                raise ValueError("appearance: one albedo per body particle and per point of every target sample")
        for i, s in enumerate(samples):
            p_t, c_t, pics = sample_pictures(s, cfg, pitch, radius, offset, lattice, h, views, extent,
                                             None if appearance is None else t_alb[i], b_ref)
            if cfg.render_target == "mesh_shell":
                refs.append(p_t)
            per.append(pics)
        e_sils, e_shade = mean([e[0] for e in per]), mean([e[1] for e in per])
        ext = Exterior(pitch, lattice, h, pitch, e_sils, e_shade, radius, offset, cfg.exterior_field,
                       albedo=b_alb, albedo_ref=b_ref, colour=getattr(cfg, "colour_kernel", "kernel"), colour_tau=0.5 * pitch)
        if appearance is not None:
            print(f"[target] appearance (D160): scene {appearance.scene}, {b_alb.shape[1]} channel(s); body albedo "
                  f"mean {[round(float(v), 4) for v in b_ref]}, target discs' (last sample) mean "
                  f"{[round(float(v), 4) for v in c_t.mean(0)]}; {appearance.note}", flush=True)
        print(f"[target] exterior: {dict(aniso='anisotropic field (D140)', zbs='Zhu-Bridson field, Solenthaler et al. correction (D144)').get(cfg.exterior_field, 'Zhu-Bridson field')}, "
              f"pitch {pitch:.4f} wu, kernel {radius:g} pitches, offset {offset:.3f}, lattice {h / pitch:.2f} pitches = "
              f"{h / (2.0 * extent / cfg.render_res):.2f} render pixels, {len(p_t)} discs on the target"
              + (f"; the pictures are the mean over {len(samples)} samples" if len(samples) > 1 else ""), flush=True)
        if cfg.render_target == "mesh_shell":
            ext.sils, ext.shade = shell_pictures(mesh, lattice, h, pitch, radius * pitch, refs, views, extent, cfg, e_sils,
                                                 e_shade)
        if getattr(cfg, "render_compare", "exterior") == "column":   # D178 (the J5 attempt): the count pictures
            from .render_loss import count_pictures
            with torch.no_grad():
                ext.counts = list(torch.stack([count_pictures(s, views, cfg.render_res, extent) for s in samples]).mean(0))
            print(f"[target] render comparison (D178): the projected count pictures of {len(samples)} render samples",
                  flush=True)
        if content is not None:               # D178 (render_split "masked"): where the pictures say more than the 3D target
            content_masks(ext, content, cfg, views, extent)
    # the W1 cleanup's fine target-fitted grid (1.5 extents each way, the box leash's range), on the target's own box
    # (D156: the render's box when the pictures are drawn from another sample)
    ext_t = extent if render_x is None else float(tgt_t.abs().max()) * 1.25
    dtdims = (cfg.dt_res,) * 3
    dtdx = 3.0 * ext_t / cfg.dt_res
    dtgmin = torch.tensor([-1.5 * ext_t] * 3, device=gpu.DEVICE)
    dt3 = target_dt_grid(target_mass_grid(tgt_t, m_t, dtgmin, dtdx, dtdims), dtdx, dtdims,
                         clamp=cfg.dt_clamp_frac * ext_t)
    knn = gpu.KNN(tgt_t)
    d1 = knn.query(tgt_t, 2)[0][:, 1]
    nn_sp = gpu.median(d1)
    m_ref, n_support = density_units(grid)
    xu = None
    if cfg.baseline.startswith("xu"):         # the baseline's target: the simulation grid, the MPM's cubic B-spline
        xu_gmin = torch.tensor(np.asarray(prm.grid_min, np.float32), device=gpu.DEVICE)
        xu_dims = (prm.nx, prm.ny, prm.nz)
        xu = (rasterize_mass_cubic(tgt_t, m_t * cfg.xu_mass, xu_gmin, prm.dx, xu_dims).detach(), xu_gmin,
              float(prm.dx), xu_dims)
    gap = gap_walls = None
    if cfg.gap_split != "off":                # D148: the target's gaps at the kernel's scale, once a run
        from ..mpm.gap_split import GapPlanes
        walls = cfg.gap_split in ("walls", "nearest", "carry")
        gp = GapPlanes(target_x if isinstance(target_x, np.ndarray) else tgt_t.cpu().numpy(), prm.dx, prm.grid_min,
                       (prm.nx, prm.ny, prm.nz), ldx, keep_boxes=walls, margin=int(walls))
        pn = gp.n.copy()
        print(f"[gap] {gp.n_gap} gap nodes ({gp.n_three} with three or more pieces) on the {prm.nx}^3 grid, the "
              f"loss cell {ldx:.4f} wu", flush=True)
        if walls:                             # D148c: each node's pieces, the particles' sides by them
            from ..mpm.gap_walls import BoxPieces
            bp = BoxPieces(gp, nearest_pieces=cfg.gap_split == "nearest")    # D148d: the nearest piece outside the box
            dev = gpu.DEVICE
            gap_walls = dict(box0=torch.as_tensor(bp.box0, device=dev), boxd=torch.as_tensor(bp.boxd, device=dev),
                             boxi=torch.as_tensor(bp.boxi, device=dev), labels=torch.as_tensor(bp.labels, device=dev),
                             occ=torch.as_tensor(bp.occ, device=dev), nearest=torch.as_tensor(bp.nearest, device=dev),
                             lo=bp.lo, cell=bp.cell, nc=bp.nc,
                             clamp={"walls": 0, "nearest": 1, "carry": 2}[cfg.gap_split])
            if cfg.gap_split == "carry":         # D148e: the walls carried between overlapping boxes
                from ..mpm.gap_walls import CarryTables
                ct = CarryTables(gp)
                gap_walls.update(gidx=ct.gidx, pairtab=ct.pairtab, pcode=ct.pcode, lmax=ct.lmax)
                print(f"[gap] carry: {ct.n_pairs} pairs of overlapping gap nodes, {ct.lmax - 1} pieces at most a box",
                      flush=True)
            print(f"[gap] pieces: {bp.n_nodes} gap nodes with their boxes' pieces (joined through the one-cell ring)",
                  flush=True)
        gap = (torch.as_tensor(gp.m, device=gpu.DEVICE), torch.as_tensor(pn, device=gpu.DEVICE))
    return TargetPack(grid=grid, lgmin=lgmin, ldx=ldx, ldims=ldims, m=m, views=views, sils=sils,
                      extent=extent, shade=shade, pgmin=lgmin, pdx=pdx, pdims=pdims, pblur=pblur,
                      dt3=dt3, dtgmin=dtgmin, dtdx=dtdx, dtdims=dtdims, pts=tgt_t, knn=knn,
                      nn_spacing=nn_sp, m_ref=m_ref, n_support=n_support, support=support, gate=gate, ext=ext,
                      xu=xu, draws=draws, gap=gap, gap_walls=gap_walls)


def calibrate_units(tgt: TargetPack, source_x: torch.Tensor, cfg: PipelineConfig) -> None:
    """Measure the legacy/density ratios of D_vol at the source: unit_ratio converts every
    fixed weight (so the relative weighting equals the legacy one at the source),
    unit_grad_ratio the gradient-magnitude constants (Adam eps, the adaptive-step norm).
    The legacy side is evaluated on the fixed reference grid (cfg.unit_ref_res), the grid
    every legacy weight was tuned on."""
    ref = int(cfg.unit_ref_res)
    if ref > 0 and ref != int(tgt.ldims[0]):
        dims_ref = (ref,) * 3
        ldx_ref = float(tgt.ldx * tgt.ldims[0] / ref)
        grid_ref = target_mass_grid(tgt.pts, tgt.m, tgt.lgmin, ldx_ref, dims_ref)
    else:
        grid_ref, ldx_ref, dims_ref = tgt.grid, tgt.ldx, tgt.ldims
    xg = source_x.detach().clone().requires_grad_(True)
    L_leg = d_vol(xg, tgt.m, grid_ref, tgt.lgmin, ldx_ref, dims_ref)
    g_leg = torch.autograd.grad(L_leg, xg)[0].norm()
    xg2 = source_x.detach().clone().requires_grad_(True)
    L_den = d_vol_density(xg2, tgt.m, tgt.grid, tgt.lgmin, tgt.ldx, tgt.ldims,
                          tgt.m_ref, tgt.n_support)
    g_den = torch.autograd.grad(L_den, xg2)[0].norm()
    L_leg, L_den = L_leg.detach(), L_den.detach()
    if float(L_den) <= 0 or float(g_den) <= 0:
        raise ValueError("the unit calibration needs a source that differs from the target")
    tgt.unit_ratio = float(L_leg / L_den)
    tgt.unit_grad_ratio = float(g_leg / g_den)


def render_floor(tgt: TargetPack, cfg: PipelineConfig, x_floor, albedo=None) -> float:
    """D162 (1): the render term (silhouette + w_pbr shading) of a sample of the render target (x_floor; the runner's
    default is the pictures' own first sample, the render target's sample or else the target's, D162 amendment 2: an
    independent sample at seed + 2 scores above the physics-only twins) against the run's pictures, read as the body's
    state is read (Objective.render_terms: the tracked exterior discs of a field found at that state, d_exterior;
    albedo (D160): the sample's own paint). What a placement of the render target's sample at the run's pitch scores:
    the pictures' floor at the run's N, resolution and views, the D156 / D158 reference."""
    from ..render.exterior import tracked
    from .render_loss import d_exterior
    e = tgt.ext
    x = gpu.tensor(x_floor)
    with torch.no_grad():
        discs = tracked(make_field(e.field, x, e.pitch, radius=e.radius, offset=e.offset, device_field=True),
                        e.lattice, e.h, e.skin, e.colour, e.colour_tau)
        if albedo is None:
            p, n, _ = discs.read(x)
            sil, pbr = d_exterior(p, n, e.sils, e.shade, tgt.views, cfg.render_res, tgt.extent, cfg.sil_k, cfg.w_hole,
                                  cfg.w_spray, cfg.pbr_ambient)
        else:
            a = gpu.tensor(np.asarray(albedo, np.float32))
            p, n, _, c = discs.read(x, a, e.albedo_ref if e.albedo_ref is not None else a.mean(0))
            sil, pbr = d_exterior(p, n, e.sils, e.shade, tgt.views, cfg.render_res, tgt.extent, cfg.sil_k, cfg.w_hole,
                                  cfg.w_spray, cfg.pbr_ambient, albedo=c)
    return float(sil + cfg.w_pbr * pbr)


def _gauss_blur(img: torch.Tensor, sigma_px: float) -> torch.Tensor:
    """Separable Gaussian blur of the last two axes (zero outside the picture), sigma in pixels."""
    import torch.nn.functional as Fn
    rad = max(1, int(np.ceil(3.0 * sigma_px)))
    k = torch.exp(-torch.arange(-rad, rad + 1, device=img.device, dtype=img.dtype) ** 2 / (2.0 * sigma_px * sigma_px))
    k = k / k.sum()
    shape = img.shape
    b = img.reshape(-1, 1, shape[-2], shape[-1])
    b = Fn.conv2d(b, k.view(1, 1, -1, 1), padding=(rad, 0))
    b = Fn.conv2d(b, k.view(1, 1, 1, -1), padding=(0, rad))
    return b.reshape(shape)


def render_term_lowpass(tgt: TargetPack, cfg: PipelineConfig, x, albedo=None, sigma=None) -> float:
    """D162d (cfg.render_weight_ref "floor_pitch"): the render term (silhouette + w_pbr shading) of the particles x as the
    body is read (the tracked exterior discs of a field found at x; with albedo, D160, the particles' paint), with both
    its pictures and the run's target pictures low-passed by a Gaussian of sigma wu (default: one exterior pitch, the
    field's own size) before they are compared by d_exterior's own measures (the asymmetric silhouette penalty, the
    shading's mean square, per colour channel). The sub-pitch arrangement of the exterior (D162 read 1, D162c) is below
    that scale, the letters' strokes (3.8 pitches) above it. sigma -> 0 reads render_floor's value (tests)."""
    from ..losses.silhouette import soft_silhouette_multi
    from ..render.exterior import tracked
    from .render_loss import shaded_discs
    e = tgt.ext
    xg = gpu.tensor(np.asarray(x.detach().cpu().numpy() if torch.is_tensor(x) else x, np.float32))
    sig_px = (e.pitch if sigma is None else float(sigma)) / (2.0 * tgt.extent / cfg.render_res)
    blur = (lambda t: t) if sig_px < 1e-3 else (lambda t: _gauss_blur(t, sig_px))  # noqa: E731
    with torch.no_grad():
        discs = tracked(make_field(e.field, xg, e.pitch, radius=e.radius, offset=e.offset, device_field=True),
                        e.lattice, e.h, e.skin, e.colour, e.colour_tau)
        if albedo is None:
            p, n, _ = discs.read(xg)
            c = None
        else:
            a = gpu.tensor(np.asarray(albedo.detach().cpu().numpy() if torch.is_tensor(albedo) else albedo, np.float32))
            p, n, _, c = discs.read(xg, a, e.albedo_ref if e.albedo_ref is not None else a.mean(0))
        al = blur(soft_silhouette_multi(p, tgt.views, cfg.render_res, tgt.extent, cfg.sil_k))
        al_t = blur(torch.stack(list(e.sils), 0))
        sil = (cfg.w_hole * torch.clamp(al_t - al, min=0.0).pow(2)
               + cfg.w_spray * torch.clamp(al - al_t, min=0.0).pow(2)).mean(dim=(1, 2)).mean()
        s, _ = shaded_discs(p, n, tgt.views, cfg.render_res, tgt.extent, cfg.sil_k, cfg.pbr_ambient, albedo=c)
        if c is None:
            pbr = (blur(s) - blur(torch.stack(list(e.shade), 0))).pow(2).mean(dim=(1, 2)).mean()
        else:
            t = torch.stack(list(e.shade), 0).movedim(1, 0).contiguous()
            s, t = blur(s), blur(t)
            pbr = sum((s[k] - t[k]).pow(2).mean(dim=(1, 2)).mean() for k in range(len(s))) / len(s)
    return float(sil + cfg.w_pbr * pbr)
