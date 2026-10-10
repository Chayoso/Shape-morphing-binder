"""Differentiable MPM rollout via per-step state arrays (wp.Tape adjoint).

Each timestep reads state t and writes state t+1, so the tape retains every
intermediate for the reverse pass. Fresh Trajectory per forward => clean tape.
Inputs may be CUDA torch tensors (the pipeline: copied on the device, no host round
trip) or numpy arrays (tests).
"""
from __future__ import annotations

import numpy as np
import torch
import warp as wp

from . import adjoints as ADJ
from . import kernels as K
from .state import MPMParams, MPMState
from .step import gate_omega, nominal_support

# D138: on a tape, the transfers' adjoints are the hand-written ones of mpm/adjoints.py (the same vector-Jacobian
# products to float rounding, a fraction of the generated ones' cost); False records the generated adjoints (tests)
HAND_ADJOINTS = True


def _is_tensor(a) -> bool:
    return type(a).__module__.startswith("torch") and hasattr(a, "is_cuda")


def to_wp(a, dtype, device="cuda", requires_grad=False):
    """A Warp array owning a copy of `a` (CUDA tensor: device-to-device; numpy: upload)."""
    if _is_tensor(a):
        src = wp.from_torch(a.detach().float().contiguous(), dtype=dtype)
        return wp.clone(src, requires_grad=requires_grad)
    return wp.array(np.ascontiguousarray(a), dtype=dtype, device=device, requires_grad=requires_grad)


def to_wp_int(a, device="cuda"):
    if _is_tensor(a):
        return wp.clone(wp.from_torch(a.detach().int().contiguous().reshape(-1), dtype=wp.int32))
    return wp.array(np.ascontiguousarray(a, np.int32).reshape(-1), dtype=wp.int32, device=device)


def scalar_or_array(val, N, device, requires_grad=False):
    """A per-particle float array from a scalar (filled on the device) or an array."""
    if isinstance(val, wp.array):
        return val
    if np.isscalar(val):
        return wp.full(N, float(val), dtype=wp.float32, device=device, requires_grad=requires_grad)
    if _is_tensor(val):
        return to_wp(val.reshape(-1), wp.float32, device, requires_grad)
    return wp.array(np.ascontiguousarray(np.broadcast_to(val, (N,)), np.float32), dtype=wp.float32,
                    device=device, requires_grad=requires_grad)


_ID_HOST: dict = {}


def _id(N):
    """(N,3,3) float32 host identities (a fresh copy of a cached array)."""
    a = _ID_HOST.get(N)
    if a is None:
        a = np.tile(np.eye(3, dtype=np.float32), (N, 1, 1))
        _ID_HOST[N] = a
    return a.copy()


_ID_CACHE: dict = {}


def _id_dev(N: int, device: str):
    """Cached device identity (N,3,3); per-step F arrays are cloned from it on the device."""
    key = (N, str(device))
    a = _ID_CACHE.get(key)
    if a is None:
        a = wp.array(_id(N), dtype=wp.mat33, device=device)      # one upload per N and process
        _ID_CACHE[key] = a
    return a


def compute_rest_volumes(x0, m, prm: MPMParams, device="cuda"):
    """Reference particle volumes Vp0, once at the sampled source state (material data, not
    rollout state: pass the result to every later Trajectory). x0: CUDA tensor -> returns a
    CUDA tensor; numpy -> numpy."""
    from .step import compute_volumes
    N = int(x0.shape[0])
    if len(x0.shape) != 2 or x0.shape[1] != 3:
        raise ValueError(f"x0 must have shape (N,3), got {tuple(x0.shape)}")

    def z(dt):
        return wp.zeros(N, dtype=dt, device=device)

    state = MPMState(x=to_wp(x0, wp.vec3, device), v=z(wp.vec3), C=z(wp.mat33),
                     F=wp.clone(_id_dev(N, device)), Fp=wp.clone(_id_dev(N, device)), dFc=z(wp.mat33),
                     P=z(wp.mat33), F_new=wp.clone(_id_dev(N, device)),
                     m=scalar_or_array(m, N, device), vol=z(wp.float32),
                     lam=z(wp.float32), mu=z(wp.float32), eta=z(wp.float32),
                     grid_m=wp.zeros(prm.ngrid, dtype=wp.float32, device=device),
                     grid_v=wp.zeros(prm.ngrid, dtype=wp.vec3, device=device), N=N, device=device)
    compute_volumes(state, prm)
    vol = wp.to_torch(state.vol).clone()
    if not bool(vol.isfinite().all()) or bool((vol < 0.0).any()):
        raise RuntimeError("rest-volume estimation produced invalid Vp0")
    return vol if _is_tensor(x0) else vol.cpu().numpy()


_WARMED: set = set()          # devices whose kernels were launched once outside a capture


def _release_idle_cuda_cache(device):
    """Leave headroom for Warp, whose allocations PyTorch cannot reclaim for.

    Only release unused cache under pressure; live tensors and captured graph
    pools stay intact. Never release during an enclosing CUDA capture.
    """
    device = str(device)
    if not device.startswith("cuda"):
        return
    import torch
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            return
        free, _ = torch.cuda.mem_get_info(device)
        idle = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
        if idle > free:
            torch.cuda.empty_cache()


class Trajectory:
    def __init__(self, x0, m, lam, mu, prm: MPMParams, T: int,
                 Fp=None, v0=None, F0=None, C0=None, dFc=None, eta=None,
                 device="cuda", requires_grad=True, mat_grad=False, vol0=None,
                 Fg0=None, track_geom=False, bonds=None, persistent=False, layer=None, layer_u=None,
                 bond_history=False, control_steps=None, polar_adjoint=False, spacing=None,
                 volume_exact=False, J0=None):
        if not _is_tensor(x0):
            x0 = np.ascontiguousarray(x0, np.float32)
        # PERSISTENT: the buffers are rolled out many times (line-search candidates); the
        # accumulated grid arrays are re-zeroed per step and the rollout can be recorded as
        # a CUDA graph (capture/run) — the same kernels with no Python launch overhead.
        self.persistent = bool(persistent)
        self.graph = None
        self.requires_grad = bool(requires_grad)
        N = int(x0.shape[0])
        self.N, self.T, self.prm, self.device = N, T, prm, device
        if control_steps is not None and (int(control_steps) != control_steps or not 1 <= control_steps <= T):
            raise ValueError("control_steps must be an integer between 1 and T")
        self.control_steps = T if control_steps is None else int(control_steps)
        # a tracked volume J: "history" (D129, True), J of the unsmoothed history, the control's volume included
        # (k_volume_update), read by the stress as (J / det F)^(1/3) (F + dFc) (k_stress_vx); "motion" (D130), the motion's
        # own, det Fg (k_volume_update_motion), read the same way; "carried" (D131), the motion's own carried in the smoothed
        # F itself (k_volume_carry: det F = J after every step), the stress reading F + dFc (the old kernels); "smoothed"
        # (D134), carried the same way, J the motion's volume at the smoothing's rate, J_t det(I + (1 - s) dt C)
        # (k_volume_update_smoothed: the old path's volume without the control's accumulated part); off (False /
        # "off"), the kernels and launches are the old path's, unchanged
        mode = {False: "off", None: "off", "": "off", True: "history"}.get(volume_exact, volume_exact)
        if mode not in ("off", "history", "motion", "carried", "smoothed"):
            raise ValueError(f"volume_exact must be off, history, motion, carried or smoothed, got {volume_exact!r}")
        self.volume_mode = mode
        self.volume_exact = mode != "off"
        self.stress_reads_J = mode in ("history", "motion")
        self.carries = mode in ("carried", "smoothed")
        if self.stress_reads_J:
            self.stress_kernel = K.k_stress_polar_vx if polar_adjoint else K.k_stress_vx
        else:
            self.stress_kernel = K.k_stress_polar if polar_adjoint else K.k_stress
        rg = requires_grad

        def A(a, dt, g=False):
            return to_wp(a, dt, device, g)

        def Z(dt, g=False):                    # device-side zeros: no host copy
            return wp.zeros(N, dtype=dt, device=device, requires_grad=g)

        def ID(g=False):                       # device-side identity clone
            return wp.clone(_id_dev(N, device), requires_grad=g)

        def AZ(a, dt, g=False):                # the given state, or zeros
            return Z(dt, g) if a is None else A(a, dt, g)

        def scratch(make, count):
            # Share only fully overwritten forward intermediates. C/Fraw must
            # retain per-step buffers: G2P skips invalid rows without writing.
            # Never alias adjoint history.
            if persistent and not rg:
                return [make()] * count
            return [make() for _ in range(count)]

        # CONTROL FIELD: a single array (one dFc shared by every step) or a list of T arrays
        # (a control SEQUENCE, dFc[t], the CompGraph formulation). `_dfc(t)` hides the difference.
        if dFc is None:
            self.dFc = Z(wp.mat33, rg)
            self.dFc_seq = None
        elif isinstance(dFc, (list, tuple)):
            assert len(dFc) == T, f"dFc sequence must have T={T} entries, got {len(dFc)}"
            self.dFc_seq = list(dFc)
            self.dFc = self.dFc_seq[0]          # volume precompute uses the t=0 control
        else:
            self.dFc = dFc
            self.dFc_seq = None
        self.release_dFc = Z(wp.mat33) if self.control_steps < T else None
        # shared, non-differentiated; material: a wp.array passes through UNCHANGED so the
        # torch bridge can hand in from_torch leaves (dL/d(lam,mu) flows back through the tape)
        self.m = scalar_or_array(m, N, device)
        self.lam = scalar_or_array(0.0 if lam is None else lam, N, device, mat_grad)
        self.mu = scalar_or_array(0.0 if mu is None else mu, N, device, mat_grad)
        self.eta = scalar_or_array(0.0 if eta is None else eta, N, device, mat_grad)
        self.Fp = ID() if Fp is None else A(Fp, wp.mat33)
        if vol0 is None:
            self.vol = Z(wp.float32)
        else:
            if tuple(vol0.shape) != (N,):
                raise ValueError(f"vol0 must have shape ({N},), got {tuple(vol0.shape)}")
            if _is_tensor(vol0):
                vol_ok = bool(vol0.isfinite().all()) and not bool((vol0 < 0).any())
            else:
                vol_ok = bool(np.isfinite(vol0).all()) and not bool((np.asarray(vol0) < 0.0).any())
            if not vol_ok:
                raise ValueError("vol0 must be finite and non-negative")
            self.vol = A(vol0, wp.float32)
        # per-step trajectory
        self.x = [A(x0, wp.vec3, rg) if t == 0 else Z(wp.vec3, rg) for t in range(T + 1)]
        self.v = [AZ(v0, wp.vec3, rg) if t == 0 else Z(wp.vec3, rg) for t in range(T + 1)]
        # C[0] must be settable: the APIC affine field is part of the state. Dropping it when a
        # trajectory is restarted from a promoted state silently discards momentum content and
        # leaves an elastically loaded body frozen -> stored energy is re-released every restart.
        self.C = [AZ(C0, wp.mat33, rg) if t == 0 else Z(wp.mat33, rg) for t in range(T + 1)]
        self.F = [(ID(rg) if F0 is None else A(F0, wp.mat33, rg)) if t == 0 else ID(rg)
                  for t in range(T + 1)]
        self.Fraw = [ID(rg) for t in range(T + 1)]
        # THE TRACKED VOLUME (D129): J[t] per step, J[0] = J0 (1 at the source; the promoted J at a window start). Per-step
        # buffers like F (the adjoint reads every step's; the trajectory check reads every step's minimum)
        if self.volume_exact:
            if J0 is not None and tuple(J0.shape) != (N,):
                raise ValueError(f"J0 must have shape ({N},), got {tuple(J0.shape)}")
            self.J = [(wp.ones(N, dtype=wp.float32, device=device, requires_grad=rg) if J0 is None
                       else A(J0, wp.float32, rg)) if t == 0 else Z(wp.float32, rg) for t in range(T + 1)]
        else:
            if J0 is not None:
                raise ValueError("J0 is the state of volume_exact; it has no meaning without it")
            self.J = None
        # D131 ("carried"): the blend of k_update lands in Fs[t+1] and k_volume_carry writes F[t+1] = (J / det Fs)^(1/3) Fs
        # (a separate buffer: an in-place write would break the tape)
        self.Fs = scratch(lambda: ID(rg), T + 1) if self.carries else None
        # GEOMETRIC deformation gradient (render kinematics; kernels.k_geom_update):
        # transported by the velocity gradient only, no control, no smoothing. Optional
        # so the physics-only paths pay nothing for it.
        self.track_geom = bool(track_geom)
        if self.track_geom:
            if Fg0 is not None and tuple(Fg0.shape) != (N, 3, 3):
                raise ValueError(f"Fg0 must have shape ({N},3,3), got {tuple(Fg0.shape)}")
            self.Fg = [(ID(rg) if Fg0 is None else A(Fg0, wp.mat33, rg)) if t == 0 else ID(rg)
                       for t in range(T + 1)]
        else:
            self.Fg = None
        self.P = scratch(lambda: Z(wp.mat33, rg), T)
        # GRID ARRAYS: the adjoint needs every step's grid (k_grid_op / k_g2p read them in
        # the backward pass), a forward-only rollout does not — step t+1 never reads step
        # t's grid, so a no-grad trajectory shares ONE set and re-zeroes it per step
        # (150k on a 233^3 grid: 7.4 GB -> 0.65 GB for the line-search trajectory).
        self.share_grid = not rg
        n_grid = 1 if self.share_grid else T
        gm_l = [wp.zeros(prm.ngrid, dtype=wp.float32, device=device, requires_grad=rg) for t in range(n_grid)]
        gmom_l = [wp.zeros(prm.ngrid, dtype=wp.vec3, device=device, requires_grad=rg) for t in range(n_grid)]
        gvel_l = [wp.zeros(prm.ngrid, dtype=wp.vec3, device=device, requires_grad=rg) for t in range(n_grid)]
        self.gm = [gm_l[t % n_grid] for t in range(T)]
        self.gmom = [gmom_l[t % n_grid] for t in range(T)]
        self.gvel = [gvel_l[t % n_grid] for t in range(T)]
        # SUPPORT-GATED APIC (kernels.k_support_gate): omega_t[p] scales the affine term m*C in
        # P2G at step t; computed outside the tape per step, read by the adjoint as a constant.
        self.omega1 = wp.ones(N, dtype=wp.float32, device=device)
        # MATERIAL RE-COUPLING of decoupled particles (kernels.k_p2g / k_update): bonds =
        # (nbr (N,K) int, rest (N,K) float, frag (N,)) frozen for this rollout; the decoupling
        # test is the 3^3-cell count of the support-gate kernels (outside the tape).
        # MINIMUM SPACING (kernels.k_update): (nbr (N,K) int, r) frozen for this rollout; r one float (every
        # particle's) or (N,) per particle (D122, a sample of two pitches); the kernel reads a pair's as the mean
        self.space_K = 0 if spacing is None else int(spacing[0].shape[1])
        self.space_r = (wp.zeros(1, dtype=wp.float32, device=device) if spacing is None
                        else scalar_or_array(spacing[1], N, device))
        self.space_nbr = None if spacing is None else to_wp_int(spacing[0], device)
        self.bonds = None
        self.bond_K = 0
        self.bond_history = bool(bond_history)
        self.nbr0 = wp.zeros(1, dtype=wp.int32, device=device)
        self.rest0 = wp.zeros(1, dtype=wp.float32, device=device)
        self.ncount0 = wp.zeros(N, dtype=wp.float32, device=device)
        if bonds is not None:
            nbr, rest, frag = bonds                  # frag: (N,) 1.0 = fragment particle
            self.bond_K = int(nbr.shape[1])
            self.bond_nbr = to_wp_int(nbr, device)
            self.bond_rest = A(rest.reshape(-1), wp.float32)
            self.bond_frag = A(frag, wp.float32)
            self.bonds = True
            # per-step decoupling test (the 3^3-cell count of the current state, outside the
            # tape, piecewise constant), OR-ed with the commit-time fragment mask
            self.cnt_b = wp.zeros(prm.ngrid, dtype=wp.int32, device=device)
            self.ncount_b = wp.zeros(N, dtype=wp.float32, device=device)
            self.omega_b = wp.ones(N, dtype=wp.float32, device=device)
            self.frag_step = wp.zeros(N, dtype=wp.float32, device=device)
            # The adjoint reads the branch taken at EACH step, not the final
            # fracture mask. Forward-only rollouts may still reuse one buffer.
            self.frag_history = ([wp.zeros(N, dtype=wp.float32, device=device) for _ in range(T)]
                                 if rg and bond_history else None)
        # OUTER-LAYER RELAXATION (kernels.k_layer_resid / k_layer_project): layer = (mask (N,),
        # nrm (N,3), nbr (N,K), w (N,K), frac[, g, depth[, ug[, ref]]]) frozen for this rollout.
        # k_update writes the advected positions into xu[t+1]; the projection writes x[t+1].
        self.layer = None
        self.layer_F = False
        if layer is not None:
            lmask, lnrm, lnbr, lw, lfrac = layer[:5]
            # P3 (kernels.k_layer_F): optional (g (N,K,3), depth), the u channel through F
            lg = layer[5] if len(layer) > 5 else None
            ldepth = float(layer[6]) if len(layer) > 6 else 0.0
            # optional per-particle gate on u (1 where u may act)
            lug = layer[7] if len(layer) > 7 else None
            self.layer_ug = wp.ones(N, dtype=wp.float32, device=device) if lug is None else A(lug, wp.float32)
            # optional per-particle reference of the relaxation (the target's own rough residual; zero without)
            lref = layer[8] if len(layer) > 8 else None
            self.layer_ref = wp.zeros(N, dtype=wp.float32, device=device) if lref is None else A(lref, wp.float32)
            self.layer_K = int(lnbr.shape[1])
            self.layer_mask = A(lmask, wp.float32)
            self.layer_nrm = A(lnrm, wp.vec3)
            self.layer_nbr = to_wp_int(lnbr, device)
            self.layer_w = A(lw.reshape(-1), wp.float32)
            self.layer_frac = float(lfrac)
            # the six rigid modes of the layer's normal displacements, n and r x n (r about the starting centre of
            # mass), and the inverse of their Gram matrix in 3x3 blocks: the relaxation loses its part along them
            # (k_layer_project, D98)
            xt = torch.as_tensor(x0, device=str(device)).double()
            mt, nt = torch.as_tensor(lmask, device=xt.device).double(), torch.as_tensor(lnrm, device=xt.device).double()
            rn = torch.linalg.cross(xt - xt.mean(0), nt, dim=1) * mt[:, None]
            modes = torch.cat((nt * mt[:, None], rn), 1)
            Minv = torch.linalg.pinv(modes.T @ modes).cpu().numpy()
            self.layer_rn = A(rn.float(), wp.vec3)
            self.layer_M = tuple(wp.mat33(*Minv[i:i + 3, j:j + 3].ravel().tolist()) for i in (0, 3) for j in (0, 3))
            self.xu = scratch(lambda: Z(wp.vec3, rg), T + 1)
            self.ld = scratch(lambda: Z(wp.float32, rg), T + 1)
            self.ls = scratch(lambda: Z(wp.float32, rg), T + 1)
            self.lb = scratch(lambda: wp.zeros(2, dtype=wp.vec3, device=device, requires_grad=rg), T + 1)
            # the position-mode control leaf u (N,): a warp view of the caller's tensor when
            # given (layer_u), else a zero buffer the eval path assigns into
            self.layer_u = (layer_u if layer_u is not None
                            else wp.zeros(N, dtype=wp.float32, device=device, requires_grad=rg))
            self.layer_frac_u = 1.0 / float(self.control_steps)
            self.release_u = Z(wp.float32) if self.control_steps < T else None
            self.layer = True
            if lg is not None:
                if self.volume_exact:
                    # P3's u through F (k_layer_F) multiplies F by I + G outside k_update; the tracked volume does not
                    # take that factor, so the two are not defined together (the settled path runs P3 off)
                    raise ValueError("volume_exact is not defined with the u channel through F (layer[5])")
                self.layer_F = True
                self.layer_g = A(lg.reshape(-1, 3), wp.vec3)
                self.layer_inv_depth = (1.0 / float(ldepth)) if ldepth > 0 else 0.0   # 0: no normal term
                self.Fu = scratch(lambda: ID(rg), T + 1)
        self.gate = bool(prm.gate_r_hi > prm.gate_r_lo)
        if self.gate:
            self.cnt = wp.zeros(prm.ngrid, dtype=wp.int32, device=device)
            self.ncount = wp.zeros(N, dtype=wp.float32, device=device)
            self.omega = [wp.ones(N, dtype=wp.float32, device=device) for t in range(T)]
            x0h = x0.detach().cpu().numpy() if _is_tensor(x0) else x0
            self.gate_n0 = float(prm.gate_n0) if prm.gate_n0 > 0 else nominal_support(x0h, prm, device)
        if vol0 is None:
            # Backward-compatible single-rollout fallback.  Production callers
            # must compute Vp0 at source initialisation and reuse it explicitly.
            self._compute_volumes()

    def _compute_volumes(self):
        prm, dev, N = self.prm, self.device, self.N
        inv_dx, gmin = 1.0 / prm.dx, wp.vec3(*prm.grid_min)
        gm = wp.zeros(prm.ngrid, dtype=float, device=dev)
        gv = wp.zeros(prm.ngrid, dtype=wp.vec3, device=dev)
        P0 = wp.zeros(N, dtype=wp.mat33, device=dev)
        v0 = wp.zeros(N, dtype=wp.vec3, device=dev)
        C0 = wp.zeros(N, dtype=wp.mat33, device=dev)
        wp.launch(K.k_stress, dim=N, inputs=[self.F[0], self.dFc, self.Fp, self.lam, self.mu, P0], device=dev)
        wp.launch(K.k_p2g, dim=N, inputs=[self.x[0], v0, C0, self.F[0], self.dFc, P0, self.m, self.vol,
                  self.omega1, self.nbr0, self.ncount0, 0, gm, gv, gmin, prm.dx, inv_dx, 0.0, 0.0, prm.nx, prm.ny, prm.nz, 0], device=dev)
        wp.launch(K.k_volume, dim=N, inputs=[self.x[0], self.m, gm, self.vol, gmin, prm.dx, inv_dx,
                  prm.nx, prm.ny, prm.nz], device=dev)

    def _omega(self, t: int):
        """Affine-transfer gate for step t (all ones unless the support gate is on)."""
        if not self.gate:
            return self.omega1
        gate_omega(self.x[t], self.prm, self.gate_n0, self.omega[t], self.ncount, self.cnt)
        return self.omega[t]

    def _bond_args(self, t: int):
        """(nbr, rest, frag, K) for step t; K = 0 (placeholders) unless bonds are attached.
        The decoupling flag is re-evaluated EVERY step from the current state (the 3^3-cell
        count, outside the tape) and OR-ed with the runner's commit-time fragment mask, so a
        particle that clears the fracture gap mid-window is bonded at once, not a window later."""
        if not self.bonds:
            return self.nbr0, self.rest0, self.ncount0, 0
        gate_omega(self.x[t], self.prm, 1.0, self.omega_b, self.ncount_b, self.cnt_b)
        frag = self.frag_step if self.frag_history is None else self.frag_history[t]
        wp.launch(K.k_frag_step, dim=self.N, inputs=[self.ncount_b, self.bond_frag, frag],
                  device=self.device)
        return self.bond_nbr, self.bond_rest, frag, self.bond_K

    def _hand_tape(self):
        """The tape the step is recorded on when the transfers take the hand-written adjoints (D138), else None."""
        if not (HAND_ADJOINTS and self.requires_grad) or self.eta.grad is not None:   # a viscosity leaf: generated
            return None
        return wp._src.context.runtime.tape

    def _dfc(self, t: int):
        """Control at step t: dFc[t] for a sequence, the shared field otherwise."""
        if t >= self.control_steps:
            return self.release_dFc
        return self.dFc if self.dFc_seq is None else self.dFc_seq[t]

    def step(self, t: int):
        prm, dev, N = self.prm, self.device, self.N
        inv_dx, gmin, fext = 1.0 / prm.dx, wp.vec3(*prm.grid_min), wp.vec3(*prm.f_ext)
        dfc = self._dfc(t)
        bnb, brest, bnc, bK = self._bond_args(t)
        if self.share_grid or self.persistent:       # P2G accumulates: fresh grid per step
            self.gm[t].zero_()
            self.gmom[t].zero_()
        if self.stress_reads_J:
            wp.launch(self.stress_kernel, dim=N, inputs=[self.F[t], dfc, self.Fp, self.J[t], self.lam, self.mu, self.P[t]],
                      device=dev)
        else:
            wp.launch(self.stress_kernel, dim=N, inputs=[self.F[t], dfc, self.Fp, self.lam, self.mu, self.P[t]], device=dev)
        omega = self._omega(t)
        tape = self._hand_tape()
        wp.launch(K.k_p2g, dim=N, inputs=[self.x[t], self.v[t], self.C[t], self.F[t], dfc, self.P[t],
                  self.m, self.vol, omega, bnb, bnc, bK, self.gm[t], self.gmom[t], gmin, prm.dx, inv_dx,
                  prm.dt, prm.drag,
                  prm.nx, prm.ny, prm.nz, int(self.bond_history)], device=dev, record_tape=tape is None)
        if tape is not None:
            ADJ.record_p2g(tape, self, t, dfc, omega, bnb, bnc, bK, N)
        wp.launch(K.k_grid_op, dim=prm.ngrid, inputs=[self.gm[t], self.gmom[t], self.gvel[t], prm.dt, fext,
                  prm.grid_min[1], prm.dx, prm.nx, prm.ny, prm.nz, prm.floor_y, prm.floor_friction,
                  K.WALL_NODES], device=dev)
        wp.launch(K.k_g2p, dim=N, inputs=[self.x[t], self.v[t + 1], self.C[t + 1], self.F[t], dfc,
                  self.Fraw[t + 1], self.gvel[t], self.eta, gmin, prm.dx, inv_dx, prm.dt, prm.nx, prm.ny, prm.nz,
                  prm.v_max, prm.eta_sym, prm.eta_mode], device=dev, record_tape=tape is None)
        if tape is not None:
            ADJ.record_g2p(tape, self, t, dfc, N)
        if self.volume_mode == "history":
            wp.launch(K.k_volume_update, dim=N, inputs=[self.F[t], self.Fraw[t + 1], self.J[t], self.J[t + 1]], device=dev)
        elif self.volume_mode == "smoothed":
            wp.launch(K.k_volume_update_smoothed, dim=N,
                      inputs=[self.C[t + 1], self.J[t], self.J[t + 1], prm.dt, prm.smoothing],
                      device=dev)
        elif self.volume_mode in ("motion", "carried"):
            wp.launch(K.k_volume_update_motion, dim=N, inputs=[self.C[t + 1], self.J[t], self.J[t + 1], prm.dt], device=dev)
        x_next = self.xu[t + 1] if self.layer else self.x[t + 1]
        if self.carries:
            F_next = self.Fs[t + 1]
        else:
            F_next = self.Fu[t + 1] if self.layer_F else self.F[t + 1]
        snbr = self.space_nbr if self.space_K > 0 else self.nbr0
        wp.launch(K.k_update, dim=N, inputs=[self.x[t], x_next, self.v[t + 1], self.F[t],
                  self.Fraw[t + 1], F_next, prm.dt, prm.smoothing,
                  bnb, brest, bnc, bK, 1.0 / float(self.control_steps),
                  snbr, self.space_K, self.space_r], device=dev, record_tape=tape is None)
        if tape is not None:
            ADJ.record_update(tape, self, t, x_next, F_next, bnb, brest, bnc, bK, snbr, N)
        if self.carries:
            wp.launch(K.k_volume_carry, dim=N, inputs=[self.Fs[t + 1], self.J[t + 1], self.F[t + 1]], device=dev)
        if self.layer:
            layer_u = self.layer_u if t < self.control_steps else self.release_u
            wp.launch(K.k_layer_resid, dim=N, inputs=[self.xu[t + 1], self.layer_mask, self.layer_nrm,
                      self.layer_nbr, self.layer_w, self.layer_K, self.ld[t + 1]], device=dev)
            self.lb[t + 1].zero_()
            wp.launch(K.k_layer_relax, dim=N, inputs=[self.ld[t + 1], self.layer_mask, self.layer_nrm, self.layer_rn,
                      self.layer_nbr, self.layer_w, self.layer_K, self.layer_frac, self.layer_ref,
                      self.ls[t + 1], self.lb[t + 1]], device=dev)
            wp.launch(K.k_layer_project, dim=N, inputs=[self.xu[t + 1], self.ls[t + 1], self.lb[t + 1], self.layer_mask,
                      self.layer_nrm, self.layer_rn, *self.layer_M, layer_u, self.layer_frac_u, self.layer_ug,
                      self.x[t + 1]], device=dev, record_tape=tape is None)
            if tape is not None:
                ADJ.record_layer_project(tape, self, t, layer_u, N)
            if self.layer_F:
                wp.launch(K.k_layer_F, dim=N, inputs=[layer_u, self.layer_ug, self.layer_mask, self.layer_nrm,
                          self.layer_nbr, self.layer_g, self.layer_K, self.layer_frac_u, self.layer_inv_depth,
                          self.Fu[t + 1], self.F[t + 1]], device=dev)
        if self.track_geom:
            wp.launch(K.k_geom_update, dim=N, inputs=[self.C[t + 1], self.Fg[t], self.Fg[t + 1],
                      prm.dt], device=dev)

    def rollout(self):
        for t in range(self.T):
            self.step(t)
        return self.x[self.T], self.F[self.T]

    def capture(self) -> bool:
        """Record the rollout as a CUDA graph (persistent trajectories on a CUDA device).
        Every buffer the graph touches lives on this object, so a replay is exactly the
        rollout of the CURRENT contents of x[0], v[0], C[0], F[0], Fg[0], J[0], the material and
        the control sequence. Modules are warmed with one plain rollout first (module
        loading is not allowed inside a capture)."""
        if not self.persistent or not str(self.device).startswith("cuda"):
            return False
        key = str(self.device)
        if key not in _WARMED:
            self.rollout()
            wp.synchronize_device(self.device)
            _WARMED.add(key)
        with wp.ScopedCapture(device=self.device) as cap:
            self.rollout()
        self.graph = cap.graph
        return True

    def run(self):
        """Roll out: replay the captured graph when there is one, else launch the kernels."""
        _release_idle_cuda_cache(self.device)
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.rollout()
        return self.x[self.T], self.F[self.T]
