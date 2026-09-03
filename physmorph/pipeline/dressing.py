"""Tier D — post-gate surface dressing solve (docs/local_global_design.md v2 §4).

Pure observation channel: per-child tangent coefficients (a,b) in the frozen
source PCA basis, optimized once per ACCEPTED commit directly against the
Gaussian image loss with no MPM adjoint in the loop. Touches nothing physical;
after the B1 split no gate, track, latch, or merit component consumes any
dressed quantity.
"""
from __future__ import annotations

import numpy as np
import torch

from ..render.children import (coeffs_to_offsets_torch, dressing_feasible_map,
                               offsets_to_coeffs, tangent_child_basis)


class DressState:
    """Owns the dressing coefficients + frozen basis + the per-frame archive.

    Archive contract (design §4.5): `frame_idx[i]` indexes `snapshots` for frame
    i of the runner's frames list — rollout intermediates carry the window's
    frozen (pre-window) dressing, the terminal frame carries the newly accepted
    dressing, held/null commits copy, outer-rejection truncation mirrors
    `del frames[k:]`. Compact: one (N,C,2) snapshot per CHANGE, ints per frame.
    """

    def __init__(self, gauss, src_x: np.ndarray, mask: np.ndarray | None,
                 cap_frac: float, dev: str):
        self.gauss = gauss
        base = gauss.source_offsets
        if base is None:
            raise ValueError("DressState requires configure_source to have run")
        t1, t2 = tangent_child_basis(src_x, mask, gauss.child_k)
        self.t1 = torch.as_tensor(t1, device=dev)
        self.t2 = torch.as_tensor(t2, device=dev)
        base_np = base.detach().cpu().numpy()
        self.coeff = torch.as_tensor(offsets_to_coeffs(base_np, t1, t2), device=dev)
        self.mask = (None if mask is None
                     else torch.as_tensor(np.asarray(mask, bool), device=dev))
        from scipy.spatial import cKDTree
        pts = src_x if mask is None else src_x[np.asarray(mask, bool)]
        self.h_src = float(np.median(cKDTree(pts).query(pts, k=2,
                                                        workers=-1)[0][:, 1]))
        self.cap = float(cap_frac) * self.h_src
        # archive
        self.snapshots = [self.coeff.detach().cpu().numpy().astype(np.float32)]
        self.frame_idx: list[int] = []
        self._cur = 0

    # ---- archive bookkeeping (design §4.5) ----
    def cover_frames(self, n_total_frames: int):
        """Assign the CURRENT dressing to every not-yet-covered frame."""
        while len(self.frame_idx) < n_total_frames:
            self.frame_idx.append(self._cur)

    def commit_snapshot(self, n_total_frames: int):
        """Window accepted + dressing re-solved: intermediates keep the frozen
        pre-window dressing, the terminal frame carries the new one."""
        while len(self.frame_idx) < n_total_frames - 1:
            self.frame_idx.append(self._cur)
        self.snapshots.append(self.coeff.detach().cpu().numpy().astype(np.float32))
        self._cur = len(self.snapshots) - 1
        self.frame_idx.append(self._cur)

    def truncate(self, n_frames: int):
        """Outer rejection: mirror `del frames[k:]` (snapshots list is append-only
        and cheap; dangling entries are unreferenced)."""
        del self.frame_idx[n_frames:]

    def offsets(self) -> torch.Tensor:
        return coeffs_to_offsets_torch(self.coeff, self.t1, self.t2)

    def export(self) -> dict:
        return {"dress_snapshots": np.stack(self.snapshots),
                "dress_frame_idx": np.asarray(self.frame_idx, np.int32),
                "dress_t1": self.t1.detach().cpu().numpy(),
                "dress_t2": self.t2.detach().cpu().numpy(),
                "dress_h_src": self.h_src}


def solve_dressing(ds: DressState, x: np.ndarray, F: np.ndarray,
                   iters: int, ls_noise_rel: float) -> dict:
    """One post-gate solve on the promoted terminal (x, F) — design §4.3.

    Line-searched Adam on the coefficients, feasibility map after every ACCEPTED
    step, energy stop rule: dL < ls_noise_rel*max(|L|,1) on two consecutive
    iterations. alpha0 = 0.1*cap (a tenth of the feasible radius — scale-derived,
    not tuned), halved on rejection within an iteration, x1.1 growth on accept.
    """
    dev = ds.coeff.device
    xt = torch.as_tensor(np.ascontiguousarray(x, np.float32), device=dev)
    Ft = torch.as_tensor(np.ascontiguousarray(F, np.float32),
                         device=dev).reshape(-1, 3, 3)
    coeff = ds.coeff.detach().clone().requires_grad_(True)

    def energy(c):
        # offsets stay FULL-N: gauss.loss/_render applies the parent mask itself
        off = coeffs_to_offsets_torch(c, ds.t1, ds.t2)
        return ds.gauss.loss(xt, Ft.reshape(-1, 9), mask=ds.mask,
                             offsets_override=off)

    with torch.no_grad():
        L0 = float(energy(coeff))
    L_pre = L0
    mom = torch.zeros_like(coeff)
    vel = torch.zeros_like(coeff)
    alpha, t, small_streak, used = 0.1 * ds.cap, 0, 0, 0
    cur = L0
    for it in range(int(iters)):
        L = energy(coeff)
        (g,) = torch.autograd.grad(L, coeff)
        if not torch.isfinite(g).all():
            break
        accepted = False
        a_try = alpha
        for _ in range(4):
            with torch.no_grad():
                t_ = t + 1
                m_ = 0.9 * mom + 0.1 * g
                v_ = 0.999 * vel + 0.001 * g * g
                step = a_try * (m_ / (1 - 0.9 ** t_)) / (
                    (v_ / (1 - 0.999 ** t_)).sqrt() + 1e-8)
                cand = dressing_feasible_map(coeff.detach() - step,
                                             ds.t1, ds.t2, Ft, ds.cap)
                Ln = float(energy(cand))
            if np.isfinite(Ln) and Ln < cur:
                with torch.no_grad():
                    coeff.copy_(cand)
                mom, vel, t = m_, v_, t_
                dL = cur - Ln
                cur = Ln
                alpha = min(a_try * 1.1, 0.1 * ds.cap)
                accepted = True
                used = it + 1
                small_streak = (small_streak + 1
                                if dL < ls_noise_rel * max(abs(cur), 1.0) else 0)
                break
            a_try *= 0.5
        if not accepted or small_streak >= 2:
            break
    with torch.no_grad():
        ds.coeff.copy_(dressing_feasible_map(coeff.detach(), ds.t1, ds.t2,
                                             Ft, ds.cap))
    return {"dress_iters_used": used, "dress_dL": L_pre - cur,
            "d_gauss_pre": L_pre, "d_gauss_post": cur}
