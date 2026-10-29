"""The separable Sinkhorn transform's axis log-sum-exp as a shared-memory CUDA kernel (D174, exact).

grid_ot._grid_logsumexp_axis (Warp) computes every output of one axis pass with one thread that reads its grid line and the
axis cost from global memory twice. This kernel computes each output with the same IEEE operations in the same order (the
line's values minus the cost, NaN returned at the first NaN, the maximum by fmaxf from -inf, a non-finite maximum returned,
then the sum of expf(value - maximum) from zero in the line's order and maximum + logf(sum)); only where the operands are
read from differs: a block loads 32 lines into shared memory once, and the axis cost (((c - j) dx)^2 / T, Toeplitz bit for
bit: GridSinkhornLoss checks it) as its first row, and each thread carries two outputs' independent chains. On the 87^3
grid 1.84x faster per transform, on the 71^3 1.21x; bit for bit against the Warp kernel on random fields with empty nodes,
empty lines and NaNs (tmp/d174/bench_lse2.py). Launched on torch's current stream, so it is captured with the solve's
CUDA graph like the Warp kernel it replaces.
"""
from __future__ import annotations

import torch

TX, TY = 32, 8
MAX_WIDTH = 300                       # shared memory (TX + 2) * width floats, under the 48 KB default

_SRC = r"""
#define NEG_INF __int_as_float(0xff800000)
__device__ __forceinline__ float one(const float* tile, int P, int tx, const float* row, int width, int c) {
    float maximum = NEG_INF;
    for (int j = 0; j < width; ++j) {
        const float value = __fsub_rn(tile[j * P + tx], row[abs(c - j)]);
        if (isnan(value)) return value;
        maximum = fmaxf(maximum, value);
    }
    if (!isfinite(maximum)) return maximum;
    float total = 0.f;
    for (int j = 0; j < width; ++j) {
        const float value = __fsub_rn(tile[j * P + tx], row[abs(c - j)]);
        total = __fadd_rn(total, expf(__fsub_rn(value, maximum)));
    }
    return __fadd_rn(maximum, logf(total));
}
extern "C" __global__ void lse_axis(const float* __restrict__ field, const float* __restrict__ cost_t, const int width,
                                    const int stride, const int numel, float* __restrict__ result) {
    extern __shared__ float sh[];
    const int TXb = blockDim.x, P = TXb + 1, line0 = blockIdx.x * TXb;
    const int lines = numel / width;
    float* row = sh + width * P;
    const int tid = threadIdx.y * TXb + threadIdx.x, nthr = TXb * blockDim.y;
    for (int k = tid; k < TXb * width; k += nthr) {         // the block's lines (a line: start + j stride, j < width)
        int tx, j;
        if (stride == 1) { tx = k / width; j = k - tx * width; } else { j = k / TXb; tx = k - j * TXb; }
        const int l = line0 + tx;
        float v = 0.f;
        if (l < lines) { const int outer = l / stride, inner = l - outer * stride;
                         v = field[(long long)outer * width * stride + inner + (long long)j * stride]; }
        sh[j * P + tx] = v;
    }
    for (int k = tid; k < width; k += nthr) row[k] = cost_t[k];   // the axis cost's first row: cost_t[c, j] = row[|c - j|]
    __syncthreads();
    const int tx = threadIdx.x, l = line0 + tx;
    if (l >= lines) return;
    const int outer = l / stride, inner = l - outer * stride;
    const long long start = (long long)outer * width * stride + inner;
    const int step = blockDim.y;
    for (int c0 = threadIdx.y; c0 < width; c0 += 2 * step) {
        const int c1 = c0 + step;
        if (c1 >= width) { result[start + (long long)c0 * stride] = one(sh, P, tx, row, width, c0); break; }
        float m0 = NEG_INF, m1 = NEG_INF;
        bool bad = false;
        for (int j = 0; j < width; ++j) {
            const float t = sh[j * P + tx];
            const float a = __fsub_rn(t, row[abs(c0 - j)]);
            const float b = __fsub_rn(t, row[abs(c1 - j)]);
            if (isnan(a) || isnan(b)) { bad = true; break; }
            m0 = fmaxf(m0, a); m1 = fmaxf(m1, b);
        }
        if (bad || !isfinite(m0) || !isfinite(m1)) {            // each output by the one-output path
            result[start + (long long)c0 * stride] = one(sh, P, tx, row, width, c0);
            result[start + (long long)c1 * stride] = one(sh, P, tx, row, width, c1);
            continue;
        }
        float s0 = 0.f, s1 = 0.f;
        for (int j = 0; j < width; ++j) {
            const float t = sh[j * P + tx];
            s0 = __fadd_rn(s0, expf(__fsub_rn(__fsub_rn(t, row[abs(c0 - j)]), m0)));
            s1 = __fadd_rn(s1, expf(__fsub_rn(__fsub_rn(t, row[abs(c1 - j)]), m1)));
        }
        result[start + (long long)c0 * stride] = __fadd_rn(m0, logf(s0));
        result[start + (long long)c1 * stride] = __fadd_rn(m1, logf(s1));
    }
}
"""
_KER = {}


def _kernel():
    if "k" not in _KER:
        import cupy as cp
        _KER["cp"] = cp
        _KER["k"] = cp.RawModule(code=_SRC, options=("-std=c++14",)).get_function("lse_axis")
    return _KER["cp"], _KER["k"]


def _view(cp, t: torch.Tensor):
    """A CuPy view of a contiguous float32 CUDA tensor (no copy, no stream handshake)."""
    mem = cp.cuda.UnownedMemory(t.data_ptr(), t.numel() * 4, t, t.device.index or 0)
    return cp.ndarray((t.numel(),), cp.float32, cp.cuda.MemoryPointer(mem, 0))


def toeplitz(cost: torch.Tensor) -> bool:
    """cost[c, j] == cost[0, |c - j|] bit for bit (the kernel reads the first row only)."""
    n = cost.shape[0]
    if cost.ndim != 2 or cost.shape[1] != n or n > MAX_WIDTH or cost.dtype != torch.float32:
        return False
    i = torch.arange(n, device=cost.device)
    return bool(torch.equal(cost.contiguous().view(torch.int32), cost[0][(i[:, None] - i[None, :]).abs()].contiguous().view(torch.int32)))


def lse_axis(field: torch.Tensor, cost_t: torch.Tensor, width: int, stride: int, result: torch.Tensor) -> None:
    """result = the axis log-sum-exp of field (flat, float32, contiguous) with the Toeplitz axis cost cost_t (width x width),
    on torch's current stream."""
    cp, ker = _kernel()
    lines = field.numel() // width
    shm = (width * (TX + 1) + width) * 4
    dev = field.device.index or 0
    with cp.cuda.Device(dev), cp.cuda.ExternalStream(torch.cuda.current_stream(field.device).cuda_stream, dev):
        ker(((lines + TX - 1) // TX,), (TX, TY), (_view(cp, field), _view(cp, cost_t), cp.int32(width), cp.int32(stride),
                                                   cp.int32(field.numel()), _view(cp, result)), shared_mem=shm)
