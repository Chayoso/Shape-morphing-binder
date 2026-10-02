"""subcell_response.py [DX=0.3024] [norelax] — D14: what surface relief the window's rollout can hold and make, by wavelength.

A slab (8 x 3 x 4 cells, free on every side, at rest, the run's material and time step) is sampled as the pipeline
samples a body (one jittered particle per voxel) at the particle spacing of a 300k run and of a 2.4M run. Its top
surface carries, or is asked to take, the relief y = A sin(2 pi x / wavelength) for wavelengths of 4, 2, 1, 0.5 and
0.25 cells. Every window is the pipeline's own rollout (physmorph.mpm.traj.Trajectory as window/setup.py builds it:
20 controlled + 20 released steps, the outer layer detected at the window start, its relaxation and the u channel,
the material bonds, the plastic assimilation between windows); there is no objective and no optimiser.
  A0 hold: the relief is in the sample from the start, no control. Per window, the relief left.
  A1 the u channel: a flat slab, u = 0.5 spacings x sin on the outer layer for one window, then two free windows.
  A2 the dFc channel: a flat slab, dFc_yy = 0.02 sin on every particle for the 20 controlled steps of one window.
The relief is the amplitude of the least-squares sinusoid at the commanded wavelength through the heights of the
top layer, away from the slab's edges: of the particles that were the top layer at the start (their displacement,
for A1 and A2) and of the layer detected in the current state."""
import math, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph import gpu                                      # noqa: E402  (before torch: CuPy binds its CUDA 12 compiler first)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
import warp as wp                                              # noqa: E402
from physmorph.mpm.constitutive import lame                    # noqa: E402
from physmorph.mpm.state import MPMParams                      # noqa: E402
from physmorph.mpm.traj import Trajectory, compute_rest_volumes  # noqa: E402
from physmorph.pipeline.config import PipelineConfig           # noqa: E402
from physmorph.pipeline.window.layer import layer_by_asymmetry, layer_relax_data, layer_spacing  # noqa: E402
from physmorph.plasticity import assimilate_elastic            # noqa: E402

dev = "cuda"
DX = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3024
RELAX = not (len(sys.argv) > 2 and sys.argv[2] == "norelax")   # the control arm: the layer relaxation switched off
cfg = PipelineConfig()
Tc, T = cfg.T, 2 * cfg.T
LX, D, LZ = 8 * DX, 3 * DX, 4 * DX
V_BODY, MASS_REF = 47.614, cfg.mass_ref_n                      # the gallery body's volume; its mass in unit particles
lam0, mu0 = lame(cfg.young, cfg.poisson)
WAVES = (4.0, 2.0, 1.0, 0.5, 0.25)


def slab(a, wave, amp, seed=97):
    """One jittered particle per voxel of pitch a under the surface y = amp sin(2 pi x / wave)."""
    rng = np.random.default_rng(seed)
    nx, ny, nz = int(round(LX / a)), int(math.ceil((D + abs(amp)) / a)) + 1, int(round(LZ / a))
    i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    c = np.stack([(i + .5) * a, -D + (j + .5) * a, (k + .5) * a], -1).reshape(-1, 3)
    c = c + rng.uniform(-.5, .5, c.shape) * a
    top = amp * np.sin(2 * math.pi * c[:, 0] / wave) if amp else 0.0
    return torch.as_tensor(c[c[:, 1] <= top].astype(np.float32), device=dev)


def params():
    g0 = (-3 * DX, -D - 3 * DX, -3 * DX)
    n = [int(math.ceil((L + 6 * DX) / DX)) + 1 for L in (LX, D + 2 * DX, LZ)]
    return MPMParams(dx=DX, nx=n[0], ny=n[1], nz=n[2], grid_min=g0)


def top_layer(x):
    """The detected outer layer's upward-facing part, away from the slab's edges."""
    mask, nrm = layer_by_asymmetry(x, layer_spacing(x))
    inner = (x[:, 0] > 1.5 * DX) & (x[:, 0] < LX - 1.5 * DX) & (x[:, 2] > DX) & (x[:, 2] < LZ - DX)
    return mask & (nrm[:, 1] > .7) & inner


def amplitude(xpos, h, wave):
    """Amplitude of the least-squares sinusoid of the given wavelength through the heights h at positions xpos."""
    kx = 2 * math.pi * xpos / wave
    A = torch.stack([torch.ones_like(kx), torch.sin(kx), torch.cos(kx)], 1).double()
    sol = torch.linalg.lstsq(A, h.double()[:, None]).solution[:, 0]
    return float((sol[1] ** 2 + sol[2] ** 2).sqrt())


class Body:
    """The promoted state of the slab between windows, as the runner keeps it."""

    def __init__(self, x, n_body):
        self.x, self.N = x, len(x)
        self.m = float(MASS_REF) / n_body                      # the dynamics mass of a particle at this resolution
        self.prm = params()
        self.vol0 = compute_rest_volumes(x, 1.0, self.prm, dev)
        self.nbr = gpu.knn(x, cfg.coh_k + 1)[1][:, 1:]
        eye = torch.eye(3, device=dev).expand(self.N, 3, 3).contiguous()
        self.F, self.Fp, self.v, self.C = eye.clone(), eye.clone(), None, None

    def window(self, u=None, dfc=None):
        """One committed window; u: (N,) normal offsets for the layer, dfc: (N,3,3) for the controlled steps.
        Returns the positions at the end of the controlled half and at the end."""
        x, N = self.x, self.N
        sp0 = layer_spacing(x)
        lmask, lnrm, lnbr, lw = layer_relax_data(x, sp0, k=cfg.layer_k, h_sp=cfg.layer_h_sp)
        dc = torch.zeros(T, N, 3, 3, device=dev)
        if dfc is not None:
            dc[:Tc] = dfc
        seq = [wp.from_torch(dc[t], dtype=wp.mat33) for t in range(T)]
        rest = (x[self.nbr] - x[:, None, :]).norm(dim=2)
        tr = Trajectory(x, self.m, lam0, mu0, self.prm, T, F0=self.F, Fp=self.Fp, v0=self.v, C0=self.C, dFc=seq, device=dev,
                        requires_grad=False, vol0=self.vol0, persistent=True, bonds=(self.nbr, rest, torch.zeros(N, device=dev)),
                        layer=(lmask, lnrm, lnbr, lw, (1.0 / float(Tc)) if RELAX else 0.0), bond_history=True, control_steps=Tc,
                        polar_adjoint=True)
        if u is not None:
            wp.to_torch(tr.layer_u).copy_((u * lmask).clamp(-sp0, sp0))
        tr.run()
        mid = wp.to_torch(tr.x[Tc]).clone()
        self.x = wp.to_torch(tr.x[T]).clone()
        self.F = wp.to_torch(tr.F[T]).reshape(N, 3, 3).clone()
        self.v, self.C = wp.to_torch(tr.v[T]).clone(), wp.to_torch(tr.C[T]).reshape(N, 3, 3).clone()
        self.Fp = assimilate_elastic(self.F, self.Fp, eta=cfg.assim, smin=cfg.assim_smin, smax=cfg.assim_smax, isochoric=True)
        return mid, self.x, sp0


print(f"cell {DX} wu; slab {LX:.2f} x {D:.2f} x {LZ:.2f} wu; window {Tc} controlled + {Tc} released steps; layer neighbours {cfg.layer_k}, weight width {cfg.layer_h_sp} spacings; "
      f"relaxation {(1 / Tc) if RELAX else 0:.3f} a step")
for label, n_body in (("300k", 300_000), ("2.4M", 2_400_000)):
    a = (V_BODY / n_body) ** (1 / 3)
    print(f"\n######## particle spacing of a {label} run: a = {a:.4f} wu = {a / DX:.3f} cells")
    print(f"== A0 hold: relief A = wavelength / 8 in the sample, no control")
    print("   wavelength (cells | spacings) | A asked (spacings) | in the sample at the start (share of asked) | left after window 1, 2, 3, 4 (share of the start; current layer) | same particles after window 4")
    for wc in WAVES:
        wave = wc * DX
        amp = wave / 8
        b = Body(slab(a, wave, amp), n_body)
        x0 = b.x.clone()
        t0 = top_layer(x0)
        a0 = amplitude(x0[t0, 0], x0[t0, 1], wave)
        left = []
        for _ in range(4):
            _, xe, sp0 = b.window()
            t = top_layer(xe)
            left.append(amplitude(xe[t, 0], xe[t, 1], wave) / a0)
        same = amplitude(x0[t0, 0], b.x[t0, 1], wave) / a0
        print(f"   {wc:5.2f} | {wave / a:5.1f} | {amp / a:5.2f} | {a0 / amp:.2f} | " + " ".join(f"{v:.2f}" for v in left) + f" | {same:.2f}   (N {b.N}, layer spacing {sp0 / a:.2f} a, top layer {int(t0.sum())})")

    print(f"== A1 the u channel: a flat slab, u = 0.5 a sin on the layer for one window, then two windows without control")
    print("   wavelength (cells | spacings) | made at the end of the controlled half | at the window's end | after free window 1, 2  (share of the commanded 0.5 a; the particles that were the top layer)")
    for wc in WAVES:
        wave = wc * DX
        b = Body(slab(a, wave, 0.0), n_body)
        x0 = b.x.clone()
        t0 = top_layer(x0)
        cmd = 0.5 * a
        u = cmd * torch.sin(2 * math.pi * x0[:, 0] / wave)
        mid, xe, _ = b.window(u=u)
        g = [amplitude(x0[t0, 0], (s - x0)[t0, 1], wave) / cmd for s in (mid, xe)]
        for _ in range(2):
            _, xe, _ = b.window()
            g.append(amplitude(x0[t0, 0], (xe - x0)[t0, 1], wave) / cmd)
        print(f"   {wc:5.2f} | {wave / a:5.1f} | {g[0]:.2f} | {g[1]:.2f} | {g[2]:.2f} {g[3]:.2f}")

    print(f"== A2 the dFc channel: a flat slab, dFc_yy = 0.02 sin on every particle for the controlled half of one window")
    print("   wavelength (cells | spacings) | surface relief at the end of the controlled half | at the window's end (spacings) | relief / wavelength against the 4-cell row (end of the controlled half | window's end)")
    base = None
    for wc in WAVES:
        wave = wc * DX
        b = Body(slab(a, wave, 0.0), n_body)
        x0 = b.x.clone()
        t0 = top_layer(x0)
        dfc = torch.zeros(b.N, 3, 3, device=dev)
        dfc[:, 1, 1] = 0.02 * torch.sin(2 * math.pi * x0[:, 0] / wave)
        mid, xe, _ = b.window(dfc=dfc)
        r = [amplitude(x0[t0, 0], (s - x0)[t0, 1], wave) for s in (mid, xe)]
        base = base or [v / wave for v in r]
        print(f"   {wc:5.2f} | {wave / a:5.1f} | {r[0] / a:.4f} | {r[1] / a:.4f} | {r[0] / wave / base[0]:.3f} | {r[1] / wave / base[1]:.3f}")
