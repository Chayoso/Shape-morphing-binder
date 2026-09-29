"""3x3 cuSOLVER Jacobi SVD with a device convergence assertion.

Matches PyTorch 2.8's small-matrix gesvdjBatched parameters and column-major
layout. Unlike linalg.svd, no convergence indices are downloaded for fallback.
Nonconvergence fails explicitly. Only called inside a custom spectral forward;
this raw helper supplies no autograd formula of its own.
"""
import torch

from physmorph.compute import cuda_module


def svd3(value):
    if not value.is_cuda:
        return torch.linalg.svd(value)
    if value.ndim != 3 or value.shape[1:] != (3, 3) or value.dtype not in (torch.float32, torch.float64):
        raise ValueError('svd3 requires (N,3,3) real float32/float64')
    if value.shape[0] < 1 or value.shape[0] > (2**31-1)//9:
        raise ValueError('svd3 batch exceeds the cuSOLVER packed-index range')
    cp = cuda_module()
    from cupy_backends.cuda.libs import cusolver
    if torch.cuda.current_device() != value.device.index:
        raise RuntimeError('svd3 requires the input CUDA device to be current')
    stream = torch.cuda.current_stream(value.device).cuda_stream
    if cp.cuda.get_current_stream().ptr != stream:
        raise RuntimeError('svd3 requires aligned Torch/CuPy streams')
    handle = cp.cuda.device.get_cusolver_handle()
    cusolver.setStream(handle, stream)
    # cuSOLVER overwrites A. Always copy, even for an already-column-major input.
    a = value.transpose(1, 2).clone(memory_format=torch.contiguous_format).transpose(1, 2)
    u, v = torch.empty_like(a), torch.empty_like(a)
    s = torch.empty((len(a), 3), device=value.device, dtype=value.dtype)
    info = torch.empty(len(a), device=value.device, dtype=torch.int32)
    prefix = 's' if value.dtype == torch.float32 else 'd'
    query = getattr(cusolver, prefix + 'gesvdjBatched_bufferSize')
    solve = getattr(cusolver, prefix + 'gesvdjBatched')
    params = cusolver.createGesvdjInfo()
    try:
        cusolver.xgesvdjSetTolerance(params, torch.finfo(value.dtype).eps)
        cusolver.xgesvdjSetMaxSweeps(params, 400)
        cusolver.xgesvdjSetSortEig(params, 1)
        args = (handle, cusolver.CUSOLVER_EIG_MODE_VECTOR, 3, 3, a.data_ptr(), 3,
                s.data_ptr(), u.data_ptr(), 3, v.data_ptr(), 3)
        size = query(*args, params, len(a))
        work = torch.empty(size, device=value.device, dtype=value.dtype)
        solve(*args, work.data_ptr(), size, info.data_ptr(), params, len(a))
        torch._assert_async((info == 0).all(), 'Assimilation SVD failed to converge')
    finally:
        cusolver.destroyGesvdjInfo(params)
    return u, s, v.transpose(1, 2)
