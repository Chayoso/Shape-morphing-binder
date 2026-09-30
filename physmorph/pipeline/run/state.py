"""The run's device-resident state between windows, its promotion from a committed window,
the fragment mask of the material bonds, and the archived trajectory.
"""
from __future__ import annotations

import torch

from ... import gpu
from ...mpm.conditioning import repair_F
from ...mpm.state import MPMParams


class FrameStore:
    """The archived trajectory: every frame's positions on the host; deformation gradients
    only where the archive keeps them (every `stride`-th frame and each window's end)."""

    def __init__(self, x0: torch.Tensor, stride: int):
        self.stride = max(1, int(stride))
        self.x = [gpu.host(x0)]
        self.F = {0: gpu.host(torch.eye(3, device=x0.device).expand(x0.shape[0], 3, 3))}

    def __len__(self):
        return len(self.x)

    def add_window(self, xs: list, Fs: list, x_end: torch.Tensor, F_end: torch.Tensor):
        """Steps 1..2T-1 of a committed window (device views) and its promoted end state."""
        for x, F in zip(xs, Fs):
            if len(self.x) % self.stride == 0:
                self.F[len(self.x)] = gpu.host(F)
            self.x.append(gpu.host(x))
        self.F[len(self.x)] = gpu.host(F_end)
        self.x.append(gpu.host(x_end))

    def hold(self):
        """One held frame at the end (the run stopped)."""
        self.F[len(self.x)] = self.F[len(self.x) - 1].copy()
        self.x.append(self.x[-1].copy())

    def truncate(self, n: int):
        del self.x[n:]
        for k in [k for k in self.F if k >= n]:
            del self.F[k]

    def archive_F(self):
        """(indices, stacked F) of every stride-th frame and the last one."""
        n = len(self.x)
        idx = sorted(set(range(0, n, self.stride)) | {n - 1})
        return idx, [self.F[i] for i in idx]


def fragment_mask(x: torch.Tensor, prm: MPMParams) -> torch.Tensor:
    """True where the particle's grid cell belongs to a connected component (26-connected,
    on the occupancy dilated by one cell: two particles couple through shared nodes of the
    4^3 stencil) that is not the largest one: material broken off the body."""
    dims = torch.tensor([prm.nx, prm.ny, prm.nz], device=x.device)
    ijk = torch.floor((x - torch.tensor(prm.grid_min, dtype=torch.float32, device=x.device)) / prm.dx).long()
    ok = ((ijk >= 0) & (ijk < dims)).all(1)
    occ = torch.zeros(prm.nx, prm.ny, prm.nz, dtype=torch.bool, device=x.device)
    io = ijk[ok]
    occ[io[:, 0], io[:, 1], io[:, 2]] = True
    lab, n = gpu.label26(gpu.dilate26(occ))
    if n <= 1:
        return torch.zeros(len(x), dtype=torch.bool, device=x.device)
    sizes = torch.bincount(lab.reshape(-1).long())
    sizes[0] = 0
    body = int(sizes.argmax())
    frag = torch.ones(len(x), dtype=torch.bool, device=x.device)
    frag[ok] = lab[io[:, 0], io[:, 1], io[:, 2]].long() != body
    return frag


def promote(commit, lo: torch.Tensor, hi: torch.Tensor):
    """The committed window's end state with the numerical pathologies repaired and counted
    (the counts must stay zero): positions clipped to the domain, F reflections and
    non-finite rows, non-finite velocities and affine fields."""
    x_new = commit.x[-1]
    n_out = int(((x_new < lo) | (x_new > hi)).any(1).sum())
    n_nan = int((~torch.isfinite(x_new).all(1)).sum())
    x = torch.clamp(torch.nan_to_num(x_new), min=lo, max=hi).float()
    F, n_bad, n_flip = repair_F(commit.end_F)
    n_ns = int((~torch.isfinite(commit.end_v).all(1)).sum()
               + (~torch.isfinite(commit.end_C).all(dim=(1, 2))).sum())
    v = torch.nan_to_num(commit.end_v).float()
    C = torch.nan_to_num(commit.end_C).float()
    counts = {"clamped": n_out, "nan_x": n_nan, "nan_state": n_ns, "F_reset": n_bad, "F_flip": n_flip,
              "F_invert_steps": int(commit.n_inv_steps)}
    return x, F.clone(), v, C, counts
