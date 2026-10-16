"""A forensic dump of the scene's committed windows (D147, the ear-tip pieces after the landing; diagnostic, off unless
scene_run.py --dump is given; it reads the window, it changes nothing). For each committed window whose start lies in
[t0, t1) and each dumped body, every step of the committed rollout (the persistent rollout's buffers hold the
commit's trajectory): positions x, velocities v (G2P's), the velocity gradient C (APIC's affine field, the grid's
velocity gradient at the particle), the stored elastic F (after the smoothing), the tracked J, the positions after the
update kernel and before the layer step (xu: what the grid, the bonds and the spacing push moved them to); and once
per window the committed controls (|dFc| per particle and driven step, u, u's gate, the layer mask) and the objective's
pull at the committed end in the body's frame (the gradients of the transport and of the cleanup with respect to the
end positions). One file per window, float32."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import warp as wp


def _np(a):
    return a.detach().float().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a, np.float32)


def window_dump(path: Path, a: int, t_start: float, opt, commit, bodies_to_dump) -> None:
    win, obj = opt.win, opt.obj
    out = {"animation": np.int64(a), "t_start": np.float64(t_start), "T": np.int64(win.T), "Tc": np.int64(win.Tc),
           "dt": np.float64(win.prm.dt), "floor_y": np.float64(win.prm.floor_y)}
    dfc = opt.leaves[0].detach()                                   # (Tc, N, 3, 3) the committed driven controls
    u = opt.u.detach()
    gate = win.u_gate()
    for b in bodies_to_dump:
        body, tr = win.bodies[b], win.tr.trajs[b]
        r = body.rows
        steps = range(win.T + 1)
        out[f"b{b}_x"] = np.stack([_np(wp.to_torch(tr.x[t])) for t in steps])
        out[f"b{b}_v"] = np.stack([_np(wp.to_torch(tr.v[t])) for t in steps])
        out[f"b{b}_C"] = np.stack([_np(wp.to_torch(tr.C[t]).reshape(-1, 3, 3)) for t in steps])
        out[f"b{b}_F"] = np.stack([_np(wp.to_torch(tr.F[t]).reshape(-1, 3, 3)) for t in steps])
        if getattr(tr, "carries", False):
            out[f"b{b}_J"] = np.stack([_np(wp.to_torch(tr.J[t])) for t in steps])
        if getattr(tr, "layer", False):
            out[f"b{b}_xu"] = np.stack([_np(wp.to_torch(tr.xu[t])) for t in range(1, win.T + 1)])
        out[f"b{b}_dfc_norm"] = _np(dfc[:, r].flatten(2).norm(dim=2))         # (Tc, n)
        out[f"b{b}_u"] = _np(u[r])
        out[f"b{b}_gate"] = _np(gate[r])
        out[f"b{b}_lmask"] = _np(win.lmask[r])
        # the objective's pull at the committed end, in the body's frame
        part = obj.parts[b]
        y = body.frame(commit.x[-1][r])[0].detach()
        for name, fn in (("transport", part.transport), ("cleanup", part.cleanup)):
            try:
                yg = y.clone().requires_grad_(True)
                (g,) = torch.autograd.grad(fn(yg), yg)
                out[f"b{b}_grad_{name}"] = _np(g)
            except Exception as e:                                 # noqa: BLE001  (a record only)
                print(f"[dump] window {a + 1} body {b}: no {name} gradient ({e!r})", flush=True)
        out[f"b{b}_y_end"] = _np(y)
        _, c, R = body.frame(commit.x[-1][r])
        out[f"b{b}_frame_c"], out[f"b{b}_frame_R"] = _np(c), _np(R)
    path.mkdir(parents=True, exist_ok=True)
    np.savez(path / f"w{a + 1:03d}.npz", **out)
