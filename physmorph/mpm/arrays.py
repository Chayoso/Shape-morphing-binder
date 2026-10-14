"""Warp arrays from torch tensors or numpy, the identity caches, the source's rest volumes and the CUDA cache release:
the helpers of the trajectory (mpm/traj.py re-exports them; moved out unchanged to keep traj.py under 500 lines)."""
from __future__ import annotations

import numpy as np
import warp as wp

from .state import MPMParams, MPMState


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
                     lam=z(wp.float32), mu=z(wp.float32),
                     grid_m=wp.zeros(prm.ngrid, dtype=wp.float32, device=device),
                     grid_v=wp.zeros(prm.ngrid, dtype=wp.vec3, device=device), N=N, device=device)
    compute_volumes(state, prm)
    vol = wp.to_torch(state.vol).clone()
    if not bool(vol.isfinite().all()) or bool((vol < 0.0).any()):
        raise RuntimeError("rest-volume estimation produced invalid Vp0")
    return vol if _is_tensor(x0) else vol.cpu().numpy()


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
