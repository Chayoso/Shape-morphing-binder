"""B13: a per-particle external force in place of the stress control dFc (cfg.baseline "force" / "force_mf").

The comparison a reviewer asks for: the same simulator, objective, render terms, optimiser and windows, with the
control variable swapped. The force f_p acts in the driven steps of a window, the steps dFc acts in, as a velocity
increment of the particle before P2G, v_p + c f_p with c = dt / m_p: P2G spreads it to the grid with the B-spline
weights, so the grid receives the impulse dt f_p w_ip (times P2G's momentum damping 1 - dt drag, which every momentum
of the step takes). The leaf is f / m_p in units of A0 = (lam + 2 mu) V / (m dx), the acceleration the P-wave
modulus's unit stress gives a particle's volume across one MPM cell (pipeline/window/setup.py), so c = dt A0 x the
leaf's scale.

"force_mf" projects each driven step's forces, at that step's positions, onto the fields with no net force and no net
torque: f'_p = f_p - a - b x (x_p - xbar), a = mean f, b = I^-1 sum (x_q - xbar) x f_q, I = sum |r|^2 1 - r r^T (equal
masses: xbar is the centre of mass). The B-spline weights reproduce constants and linear functions (sum_i w_ip = 1,
sum_i w_ip x_i = x_p), so the grid receives sum_i dt f'_i = 0 and sum_i x_i x dt f'_i = dt sum_p x_p x f'_p = 0: the
momentum-free force injects neither linear nor angular momentum. It is still not the divergence of a stress.

The moments are summed in float64 (bins, then a few threads); the forward and the adjoint are hand-written and recorded
on the tape (wp.Tape.record_func), the adjoint of the projection including its dependence on the positions.
tests/test_force_baseline.py checks the adjoint against float64 autograd and a window against finite differences.
"""
from __future__ import annotations

import sys

import torch
import warp as wp

_LOADED: set = set()           # devices this module's kernels were loaded on (force.attach)
_NB = 4096                    # bins the per-particle moments are added into (p % _NB)
_BSUM = 64                     # threads that sum the bins
_NM = 15                       # forward moments: sum x (3), sum f (3), sum x x f (3), sum x x^T (6)
_NA = 6                        # adjoint moments: sum g (3), sum g x x (3)
_NST = 18                      # a step's statistics: xbar (3), a (3), b (3), I^-1 (9)
_NAST = 9                      # the adjoint's: G (3), beta (3), R (3)


@wp.func
def _d(v: wp.vec3):
    return wp.vec3d(wp.float64(v[0]), wp.float64(v[1]), wp.float64(v[2]))


@wp.func
def _f(v: wp.vec3d):
    return wp.vec3(wp.float32(v[0]), wp.float32(v[1]), wp.float32(v[2]))


@wp.func
def _v3(a: wp.array(dtype=wp.float64), k: int):
    return wp.vec3d(a[k], a[k + 1], a[k + 2])


@wp.kernel(enable_backward=False)
def k_kick(v: wp.array(dtype=wp.vec3), f: wp.array(dtype=wp.vec3), c: float, vk: wp.array(dtype=wp.vec3)):
    p = wp.tid()
    vk[p] = v[p] + c * f[p]


@wp.kernel(enable_backward=False)
def k_kick_adj(adj_vk: wp.array(dtype=wp.vec3), c: float, adj_v: wp.array(dtype=wp.vec3),
               adj_f: wp.array(dtype=wp.vec3), has_adj_f: int):
    p = wp.tid()
    a = adj_vk[p]
    adj_v[p] = adj_v[p] + a
    if has_adj_f != 0:
        adj_f[p] = adj_f[p] + c * a


@wp.kernel(enable_backward=False)
def k_mf_moments(x: wp.array(dtype=wp.vec3), f: wp.array(dtype=wp.vec3), bins: wp.array(dtype=wp.float64)):
    p = wp.tid()
    xd = _d(x[p])
    fd = _d(f[p])
    c = wp.cross(xd, fd)
    k = (p % _NB) * _NM
    for j in range(3):
        wp.atomic_add(bins, k + j, xd[j])
        wp.atomic_add(bins, k + 3 + j, fd[j])
        wp.atomic_add(bins, k + 6 + j, c[j])
    wp.atomic_add(bins, k + 9, xd[0] * xd[0])
    wp.atomic_add(bins, k + 10, xd[1] * xd[1])
    wp.atomic_add(bins, k + 11, xd[2] * xd[2])
    wp.atomic_add(bins, k + 12, xd[0] * xd[1])
    wp.atomic_add(bins, k + 13, xd[0] * xd[2])
    wp.atomic_add(bins, k + 14, xd[1] * xd[2])


@wp.kernel(enable_backward=False)
def k_bin_sums(bins: wp.array(dtype=wp.float64), nm: int, out: wp.array(dtype=wp.float64)):
    """_BSUM threads, each the sum of _NB / _BSUM bins of every moment, added into out (zeroed before)."""
    t = wp.tid()
    for j in range(nm):
        s = wp.float64(0.0)
        for i in range(_NB // _BSUM):
            s = s + bins[(t * (_NB // _BSUM) + i) * nm + j]
        wp.atomic_add(out, j, s)


@wp.kernel(enable_backward=False)
def k_mf_finalize(s: wp.array(dtype=wp.float64), n: int, st: wp.array(dtype=wp.float64)):
    """xbar, a = mean f, b = I^-1 tau and I^-1 from the raw moments (one thread)."""
    N = wp.float64(n)
    xb = _v3(s, 0) / N
    Sf = _v3(s, 3)
    tau = _v3(s, 6) - wp.cross(xb, Sf)
    S = wp.mat33d(s[9], s[12], s[13], s[12], s[10], s[14], s[13], s[14], s[11]) - N * wp.outer(xb, xb)
    I = (S[0, 0] + S[1, 1] + S[2, 2]) * wp.identity(n=3, dtype=wp.float64) - S
    Ii = wp.inverse(I)
    b = Ii @ tau
    a = Sf / N
    for j in range(3):
        st[j] = xb[j]
        st[3 + j] = a[j]
        st[6 + j] = b[j]
        for k in range(3):
            st[9 + 3 * j + k] = Ii[j, k]


@wp.kernel(enable_backward=False)
def k_mf_apply(x: wp.array(dtype=wp.vec3), f: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
               st: wp.array(dtype=wp.float64), c: float, fo: wp.array(dtype=wp.vec3), vk: wp.array(dtype=wp.vec3)):
    p = wp.tid()
    r = _d(x[p]) - _v3(st, 0)
    g = _f(_d(f[p]) - _v3(st, 3) - wp.cross(_v3(st, 6), r))
    fo[p] = g
    vk[p] = v[p] + c * g


@wp.kernel(enable_backward=False)
def k_mf_adj_moments(x: wp.array(dtype=wp.vec3), adj_vk: wp.array(dtype=wp.vec3), c: float,
                     bins: wp.array(dtype=wp.float64), adj_v: wp.array(dtype=wp.vec3)):
    p = wp.tid()
    a = adj_vk[p]
    adj_v[p] = adj_v[p] + a
    g = wp.float64(c) * _d(a)
    gx = wp.cross(g, _d(x[p]))
    k = (p % _NB) * _NA
    for j in range(3):
        wp.atomic_add(bins, k + j, g[j])
        wp.atomic_add(bins, k + 3 + j, gx[j])


@wp.kernel(enable_backward=False)
def k_mf_adj_finalize(s: wp.array(dtype=wp.float64), n: int, st: wp.array(dtype=wp.float64),
                      ast: wp.array(dtype=wp.float64)):
    """G = sum g, H = sum g x r = sum g x x - G x xbar, beta = I^-1 H, R = sum of the r-adjoints = b x G + (N a) x beta
    (the terms in sum r vanish)."""
    N = wp.float64(n)
    G = _v3(s, 0)
    xb = _v3(st, 0)
    Ii = wp.mat33d(st[9], st[10], st[11], st[12], st[13], st[14], st[15], st[16], st[17])
    beta = Ii @ (_v3(s, 3) - wp.cross(G, xb))
    R = wp.cross(_v3(st, 6), G) + wp.cross(N * _v3(st, 3), beta)
    for j in range(3):
        ast[j] = G[j]
        ast[3 + j] = beta[j]
        ast[6 + j] = R[j]


@wp.kernel(enable_backward=False)
def k_mf_adj_apply(x: wp.array(dtype=wp.vec3), f: wp.array(dtype=wp.vec3), adj_vk: wp.array(dtype=wp.vec3), c: float,
                   n: int, st: wp.array(dtype=wp.float64), ast: wp.array(dtype=wp.float64),
                   adj_f: wp.array(dtype=wp.vec3), has_adj_f: int, adj_x: wp.array(dtype=wp.vec3)):
    p = wp.tid()
    inv_n = wp.float64(1.0) / wp.float64(n)
    g = wp.float64(c) * _d(adj_vk[p])
    r = _d(x[p]) - _v3(st, 0)
    b = _v3(st, 6)
    beta = _v3(ast, 3)
    if has_adj_f != 0:
        adj_f[p] = adj_f[p] + _f(g - inv_n * _v3(ast, 0) + wp.cross(beta, r))
    ar = (wp.cross(_d(f[p]), beta) - wp.float64(2.0) * wp.dot(beta, b) * r + wp.dot(r, b) * beta
          + wp.dot(beta, r) * b + wp.cross(b, g))
    adj_x[p] = adj_x[p] + _f(ar - inv_n * _v3(ast, 6))


_DUMMY: dict = {}


def _dummy_vec(device):
    a = _DUMMY.get(str(device))
    if a is None:
        a = wp.zeros(1, dtype=wp.vec3, device=device)
        _DUMMY[str(device)] = a
    return a


def attach(tr, force) -> None:
    """force = (f, mf, c): f a list of control_steps (N,) vec3 arrays (the leaf, a view), mf the momentum-free
    projection, c the velocity increment of a unit leaf over one step. Per-step buffers on a tape, shared without."""
    f, mf, c = force
    if len(f) != tr.control_steps:
        raise ValueError(f"the force needs one array per driven step ({tr.control_steps}), got {len(f)}")
    dev, N, rg = tr.device, tr.N, tr.requires_grad
    if str(dev) not in _LOADED:                # this module's kernels, loaded before any graph capture launches them
        wp.load_module(module=sys.modules[__name__], device=dev)
        _LOADED.add(str(dev))
    steps = tr.control_steps if rg else 1

    def per_step(make):
        one = [make() for _ in range(steps)]
        return [one[t % steps] for t in range(tr.control_steps)]

    tr.force_f = list(f)
    tr.force_mf = bool(mf)
    tr.force_c = float(c)
    tr.force_vk = per_step(lambda: wp.zeros(N, dtype=wp.vec3, device=dev, requires_grad=rg))
    if tr.force_mf:
        tr.force_fo = per_step(lambda: wp.zeros(N, dtype=wp.vec3, device=dev))      # the projected forces (a record)
        tr.force_st = per_step(lambda: wp.zeros(_NST, dtype=wp.float64, device=dev))
        tr.force_bins = wp.zeros(_NB * _NM, dtype=wp.float64, device=dev)
        tr.force_sums = wp.zeros(_NM, dtype=wp.float64, device=dev)
        if rg:
            tr.force_abins = wp.zeros(_NB * _NA, dtype=wp.float64, device=dev)
            tr.force_asums = wp.zeros(_NA, dtype=wp.float64, device=dev)
            tr.force_ast = wp.zeros(_NAST, dtype=wp.float64, device=dev)


def kick(tr, t: int):
    """The velocity P2G reads at driven step t: v[t] + c f'[t] (f' = f, or its momentum-free projection at x[t]);
    its adjoint recorded on the active tape."""
    dev, N, c = tr.device, tr.N, tr.force_c
    x, v, f, vk = tr.x[t], tr.v[t], tr.force_f[t], tr.force_vk[t]
    if tr.force_mf:
        st, fo = tr.force_st[t], tr.force_fo[t]
        tr.force_bins.zero_()
        tr.force_sums.zero_()
        wp.launch(k_mf_moments, dim=N, inputs=[x, f, tr.force_bins], device=dev, record_tape=False)
        wp.launch(k_bin_sums, dim=_BSUM, inputs=[tr.force_bins, _NM, tr.force_sums], device=dev, record_tape=False)
        wp.launch(k_mf_finalize, dim=1, inputs=[tr.force_sums, N, st], device=dev, record_tape=False)
        wp.launch(k_mf_apply, dim=N, inputs=[x, f, v, st, c, fo, vk], device=dev, record_tape=False)
    else:
        wp.launch(k_kick, dim=N, inputs=[v, f, c, vk], device=dev, record_tape=False)
    tape = wp._src.context.runtime.tape if tr.requires_grad else None
    if tape is not None:
        has_f = f.grad is not None
        a_f = f.grad if has_f else _dummy_vec(dev)
        if tr.force_mf:
            ab, asum, ast = tr.force_abins, tr.force_asums, tr.force_ast

            def backward():
                ab.zero_()
                asum.zero_()
                wp.launch(k_mf_adj_moments, dim=N, inputs=[x, vk.grad, c, ab, v.grad], device=dev)
                wp.launch(k_bin_sums, dim=_BSUM, inputs=[ab, _NA, asum], device=dev)
                wp.launch(k_mf_adj_finalize, dim=1, inputs=[asum, N, st, ast], device=dev)
                wp.launch(k_mf_adj_apply, dim=N, inputs=[x, f, vk.grad, c, N, st, ast, a_f, int(has_f), x.grad],
                          device=dev)
        else:
            def backward():
                wp.launch(k_kick_adj, dim=N, inputs=[vk.grad, c, v.grad, a_f, int(has_f)], device=dev)
        tape.record_func(backward=backward, arrays=[a for a in (x, v, vk, f) if a.grad is not None])
    return vk


def rigid_free(f: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """The momentum-free projection in float64 (torch; the reference of the tests and the records): f - a - b x r,
    r = x - mean x, a = mean f, b = I^-1 sum r x f."""
    f, x = f.double(), x.double()
    r = x - x.mean(0)
    tau = torch.linalg.cross(r, f, dim=1).sum(0)
    S = r.T @ r
    I = torch.trace(S) * torch.eye(3, dtype=S.dtype, device=S.device) - S
    b = torch.linalg.solve(I, tau)
    return f - f.mean(0) - torch.linalg.cross(b.expand_as(r), r, dim=1)


def impulse_record(f: torch.Tensor, xs, c: float, mf: bool, c0: torch.Tensor) -> dict:
    """A window's record of the committed force (equal masses; per unit mass, before P2G's damping): the centre of
    mass's velocity increment over the driven steps (force_dv, wu/s; force_vdv the vector), the angular momentum per
    unit mass it added about the window's starting centre of mass c0 (force_dL, force_vdL), the mean particle's gross
    increment (force_gross) and the leaf's largest component (force_absmax). xs: the driven steps' positions."""
    dv = torch.zeros(3, dtype=torch.float64, device=f.device)
    dL = torch.zeros(3, dtype=torch.float64, device=f.device)
    gross = 0.0
    for t, x in enumerate(xs):
        inc = c * (rigid_free(f[t], x) if mf else f[t].double())
        dv += inc.mean(0)
        dL += torch.linalg.cross(x.double() - c0.double(), inc, dim=1).mean(0)
        gross += float(inc.norm(dim=1).mean())
    return {"force_dv": float(dv.norm()), "force_vdv": dv.tolist(), "force_dL": float(dL.norm()),
            "force_vdL": dL.tolist(), "force_gross": gross, "force_absmax": float(f.abs().max())}
