"""3x3 cuSOLVER Jacobi SVD with a device convergence assertion.

Matches PyTorch 2.8's small-matrix gesvdjBatched parameters and column-major
layout. Unlike linalg.svd, no convergence indices are downloaded for fallback.
Nonconvergence fails explicitly. Only called inside a custom spectral forward;
this raw helper supplies no autograd formula of its own.
"""
import ctypes
from functools import lru_cache

import torch

from physmorph.compute import cuda_module


@lru_cache(maxsize=1)
def _solver_library():
    # hyde06 CUDA12 uses this ABI. CuPy14's solver wrappers unconditionally
    # reject capture; call the same installed library with typed C arguments.
    lib = ctypes.CDLL('libcusolver.so.11')
    ptr, integer = ctypes.c_void_p, ctypes.c_int
    declarations = {
        'cusolverDnSetStream': [ptr, ptr],
        'cusolverDnCreateGesvdjInfo': [ctypes.POINTER(ptr)],
        'cusolverDnDestroyGesvdjInfo': [ptr],
        'cusolverDnXgesvdjSetTolerance': [ptr, ctypes.c_double],
        'cusolverDnXgesvdjSetMaxSweeps': [ptr, integer],
        'cusolverDnXgesvdjSetSortEig': [ptr, integer],
    }
    common = [ptr, integer, integer, integer, ptr, integer, ptr, ptr, integer, ptr, integer]
    for prefix in ('S', 'D'):
        declarations[f'cusolverDn{prefix}gesvdjBatched_bufferSize'] = common + [ctypes.POINTER(integer), ptr, integer]
        declarations[f'cusolverDn{prefix}gesvdjBatched'] = common + [ptr, integer, ptr, ptr, integer]
    for name, signature in declarations.items():
        func = getattr(lib, name)
        func.argtypes, func.restype = signature, integer
    return lib


def _call(function, *args):
    status = function(*args)
    if status != 0:
        raise RuntimeError(f'{function.__name__} failed with cuSOLVER status {status}')


def svd3(value):
    if not value.is_cuda:
        return torch.linalg.svd(value)
    if value.ndim != 3 or value.shape[1:] != (3, 3) or value.dtype not in (torch.float32, torch.float64):
        raise ValueError('svd3 requires (N,3,3) real float32/float64')
    if value.shape[0] < 1 or value.shape[0] > (2**31-1)//9:
        raise ValueError('svd3 batch exceeds the cuSOLVER packed-index range')
    cp = cuda_module()
    if torch.cuda.current_device() != value.device.index:
        raise RuntimeError('svd3 requires the input CUDA device to be current')
    stream = torch.cuda.current_stream(value.device).cuda_stream
    if cp.cuda.get_current_stream().ptr != stream:
        raise RuntimeError('svd3 requires aligned Torch/CuPy streams')
    handle = cp.cuda.device.get_cusolver_handle()
    solver = _solver_library()
    _call(solver.cusolverDnSetStream, handle, stream)
    # cuSOLVER overwrites A. Always copy, even for an already-column-major input.
    a = value.transpose(1, 2).clone(memory_format=torch.contiguous_format).transpose(1, 2)
    u, v = torch.empty_like(a), torch.empty_like(a)
    s = torch.empty((len(a), 3), device=value.device, dtype=value.dtype)
    info = torch.empty(len(a), device=value.device, dtype=torch.int32)
    prefix = 'S' if value.dtype == torch.float32 else 'D'
    query = getattr(solver, 'cusolverDn' + prefix + 'gesvdjBatched_bufferSize')
    solve = getattr(solver, 'cusolverDn' + prefix + 'gesvdjBatched')
    params = ctypes.c_void_p()
    _call(solver.cusolverDnCreateGesvdjInfo, ctypes.byref(params))
    try:
        _call(solver.cusolverDnXgesvdjSetTolerance, params, torch.finfo(value.dtype).eps)
        _call(solver.cusolverDnXgesvdjSetMaxSweeps, params, 400)
        _call(solver.cusolverDnXgesvdjSetSortEig, params, 1)
        args = (handle, 1, 3, 3, a.data_ptr(), 3,
                s.data_ptr(), u.data_ptr(), 3, v.data_ptr(), 3)
        size = ctypes.c_int()
        _call(query, *args, ctypes.byref(size), params, len(a))
        work = torch.empty(size.value, device=value.device, dtype=value.dtype)
        _call(solve, *args, work.data_ptr(), size.value, info.data_ptr(), params, len(a))
        torch._assert_async((info == 0).all(), 'Assimilation SVD failed to converge')
    finally:
        _call(solver.cusolverDnDestroyGesvdjInfo, params)
    return u, s, v.transpose(1, 2)
