"""Window telemetry: how the render and physics channels steer the accepted step, and the
optional gradient dump (per-particle covectors, their pull-back to the controls, and
linear-response rollouts of each channel alone) that the gradient videos are made from.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import warp as wp

from ... import gpu


def linearized_work(grads, deltas) -> tuple[float, list[float]]:
    """Total and per-state -grad . delta endpoint work. Dominated by loss scale; use
    steer_cos for steering."""
    parts = [(-float((g.detach() * d.detach()).sum()) if g is not None else 0.0)
             for g, d in zip(grads, deltas)]
    return sum(parts), parts


def steer_cos(grads, deltas) -> float | None:
    """Cosine between the accepted state step and a channel's descent direction (-grad),
    joint over the given state slots."""
    num = den_g = den_d = 0.0
    for g, d in zip(grads, deltas):
        if g is None:
            continue
        g, d = g.detach(), d.detach()
        num += -float((g * d).sum())
        den_g += float((g * g).sum())
        den_d += float((d * d).sum())
    if den_g <= 0 or den_d <= 0:
        return None
    return num / (den_g ** 0.5 * den_d ** 0.5)


def work_record(phys_diag, rend_diag, state, state_new) -> dict:
    """Steering of the accepted step by each channel: endpoint gradients (first and last
    iteration) against the state change of the accepted candidate."""
    dx = state_new[0] - state[0].detach()
    dF = state_new[1] - state[1].detach()
    dv = state_new[2] - state[2].detach()
    pw, (pwx, pwF, pwv) = linearized_work(phys_diag, (dx, dF, dv))
    out = {"phys_work": pw, "phys_work_x": pwx, "phys_work_F": pwF, "phys_work_v": pwv,
           "phys_cos": steer_cos(phys_diag, (dx, dF, dv)),
           "render_work": None, "render_work_x": None, "render_work_F": None, "render_cos": None}
    if rend_diag is not None:
        rw, (rwx, rwF) = linearized_work(rend_diag, (dx, dF))
        out.update({"render_work": rw, "render_work_x": rwx, "render_work_F": rwF,
                    "render_cos": steer_cos(rend_diag, (dx, dF))})
    return out


def collect_grad_dump(e, Lp_core, gp, leaves, u, w_pbr) -> dict:
    """First-iteration covectors of each channel on the particles and their pull-back to
    the control leaves (the graph is still retained here)."""
    lsil = e.lr - w_pbr * e.lpbr
    xT = e.xT

    def g_of(term, wrt):
        g = torch.autograd.grad(term, wrt, retain_graph=True, allow_unused=True)
        return g if isinstance(wrt, (list, tuple)) else g[0]

    def gu(term):
        g = torch.autograd.grad(term, u, retain_graph=True, allow_unused=True)[0]
        return torch.zeros_like(u) if g is None else g.detach().clone()

    return {"gx_phys": g_of(Lp_core, xT), "gx_sil": g_of(lsil, xT), "gx_pbr": g_of(e.lpbr, xT),
            "gl_phys": gp[0].detach().clone(), "gl_sil": g_of(lsil, leaves)[0],
            "gl_pbr": g_of(e.lpbr, leaves)[0], "gl_rend": g_of(e.lr, leaves)[0],
            "xT0": xT.detach().clone(), "gu_phys": gp[-1].detach().clone(), "gu_sil": gu(lsil),
            "gu_pbr": gu(e.lpbr), "gu_rend": gu(e.lr)}


def write_grad_dump(directory, dump, win, leaf0, leaf_final, u_final, commit_dc, frame_end):
    """Linear-response rollouts — each channel's control gradient alone, scaled to the norm
    of the accepted change of this window — and the dump file. The committed rollout is
    restored in the trajectory buffers afterwards."""
    tr, T, N = win.tr, win.T, win.N
    step = float((leaf_final - leaf0).norm())
    resp, red, u_red = {}, {}, {}

    def end_positions():
        tr.run()
        return gpu.host(wp.to_torch(tr.x[T]).clone())

    for name in ("gl_phys", "gl_sil", "gl_pbr", "gl_rend"):
        gl = dump.get(name)
        if gl is None:
            continue
        g4 = win.expand(gl).detach().view(T, N, 9)
        red[name + "_pnorm"] = gpu.host(g4.norm(dim=(0, 2)))
        red[name + "_tmean"] = gpu.host(g4.mean(0))
        gn = float(gl.norm())
        if gn > 0 and step > 0:
            win.dc_buf.copy_(win.expand(leaf0 - (step / gn) * gl).view(T, N, 3, 3))
            resp["xT_" + name[3:]] = end_positions()
    win.dc_buf.copy_(win.expand(leaf0).view(T, N, 3, 3))
    resp["xT_base"] = end_positions()
    layer_u = wp.to_torch(tr.layer_u)
    u_step = float(u_final.norm())
    for name in ("gu_phys", "gu_sil", "gu_pbr", "gu_rend"):
        g = dump[name]
        u_red[name] = gpu.host(g)
        gn = float(g.norm())
        if gn > 0 and u_step > 0:
            layer_u.copy_((-(u_step / gn) * g).clamp(-win.sp0, win.sp0))
            resp["xT_" + name[3:] + "_u"] = end_positions()
    layer_u.zero_()
    resp["xT_base_u0"] = end_positions()
    u_red.update(u_final=gpu.host(u_final), u_step=u_step, layer_mask=gpu.host(win.lmask))
    layer_u.copy_(u_final)
    win.dc_buf.copy_(commit_dc.view(T, N, 3, 3))
    tr.run()
    os.makedirs(directory, exist_ok=True)
    k_win = len([f for f in os.listdir(directory) if f.startswith("win_")])
    np.savez_compressed(
        os.path.join(directory, f"win_{k_win:04d}.npz"), x0=gpu.host(win.x0),
        xT0=gpu.host(dump["xT0"]), xT_final=frame_end, gx_phys=gpu.host(dump["gx_phys"]),
        gx_sil=gpu.host(dump["gx_sil"]), gx_pbr=gpu.host(dump["gx_pbr"]), lam_r=dump["lam_r"],
        g_share=dump["g_share"], step_norm=step, leaf0_norm=float(leaf0.norm()),
        leaf_final_norm=float(leaf_final.norm()), **red, **resp, **u_red)


def support_record(tgt, horizon: float, x: torch.Tensor, v: torch.Tensor) -> dict:
    """The transport term taken apart at a committed state: the transport without the support
    bound E, the support penalty B, the support-gradient weight w (E / (E + w B))^2 that every
    particle feels through the bound, and the per-particle penalty's max, p99 and median."""
    sup, ot = tgt.support, tgt.grid_ot
    if sup is None or ot is None:
        return {}
    with torch.no_grad():
        saved, ot.support = ot.support, None
        try:
            E = float(ot.state_energy(x, tgt.m, v, horizon))
        finally:
            ot.support = saved
        pen = sup.penalty_per_point(x).double()
        q = torch.quantile(pen, torch.tensor([.99, .5], dtype=pen.dtype, device=pen.device))
    B, w = float(pen.mean()), sup.weight
    B_t = None
    if sup.two_sided:
        with torch.no_grad():
            B_t = float(sup.target_penalty_per_point(x).double().mean())
        B = .5 * (B + B_t)                                       # the bound sees the mean of the two sides
    w_eff = w * (E / (E + w * B)) ** 2 if np.isfinite(E) and E + w * B > 0 else None
    return {"sup_E": E, "sup_B": B, "sup_B_target": B_t, "sup_w_eff": w_eff, "sup_pen_max": float(pen.max()),
            "sup_pen_p99": float(q[0]), "sup_pen_med": float(q[1])}
