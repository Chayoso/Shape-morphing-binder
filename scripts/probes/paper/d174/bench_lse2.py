"""bench_lse2.py N [REPS] -- D174: the Sinkhorn transform's axis log-sum-exp on an N^3 grid, the production Warp kernel
against shared-memory CUDA kernels that compute every output with the same IEEE operations in the same order:
  v1: the line in shared memory, the cost row from the matrix;
  v2: + the axis cost as its first row (the matrix is Toeplitz bit for bit: ((c - j) dx)^2 / temp depends on |c - j|);
  v3: + two outputs a thread (two independent chains).
Bit-for-bit check on random fields with -inf nodes, empty lines and NaNs; times per transform (three axes)."""
import sys
import time

import physmorph  # noqa: F401
import torch
import warp as wp
import cupy as cp

from physmorph.losses.grid_ot import _grid_logsumexp_axis

n = int(sys.argv[1])
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 200
dev = "cuda"
dims = (n, n, n)
dx = 0.1
costs = [((torch.arange(m, device=dev, dtype=torch.float32)[:, None]
           - torch.arange(m, device=dev, dtype=torch.float32)[None, :]) * dx).square() for m in dims]

COMMON = r"""
#define NEG_INF __int_as_float(0xff800000)
__device__ __forceinline__ void load_tile(const float* __restrict__ field, float* tile, int width, int stride, int numel,
                                          int line0, int TX, int P) {
    const int lines = numel / width;
    const int tid = threadIdx.y * TX + threadIdx.x, nthr = TX * blockDim.y;
    for (int k = tid; k < TX * width; k += nthr) {
        int tx, j;
        if (stride == 1) { tx = k / width; j = k - tx * width; } else { j = k / TX; tx = k - j * TX; }
        const int l = line0 + tx;
        float v = 0.f;
        if (l < lines) { const int outer = l / stride, inner = l - outer * stride;
                         v = field[(long long)outer * width * stride + inner + (long long)j * stride]; }
        tile[j * P + tx] = v;
    }
}
__device__ __forceinline__ float one(const float* tile, int P, int tx, const float* crow, int width, int c, int toe) {
    float maximum = NEG_INF;
    for (int j = 0; j < width; ++j) {
        const float value = __fsub_rn(tile[j * P + tx], toe ? crow[abs(c - j)] : crow[j]);
        if (isnan(value)) return value;
        maximum = fmaxf(maximum, value);
    }
    if (!isfinite(maximum)) return maximum;
    float total = 0.f;
    for (int j = 0; j < width; ++j) {
        const float value = __fsub_rn(tile[j * P + tx], toe ? crow[abs(c - j)] : crow[j]);
        total = __fadd_rn(total, expf(__fsub_rn(value, maximum)));
    }
    return __fadd_rn(maximum, logf(total));
}
"""
SRC = COMMON + r"""
extern "C" __global__ void v1(const float* __restrict__ field, const float* __restrict__ cost_t, const int width,
                              const int stride, const int numel, float* __restrict__ result) {
    extern __shared__ float sh[];
    const int TX = blockDim.x, P = TX + 1, line0 = blockIdx.x * TX;
    load_tile(field, sh, width, stride, numel, line0, TX, P);
    __syncthreads();
    const int tx = threadIdx.x, l = line0 + tx;
    if (l >= numel / width) return;
    const int outer = l / stride, inner = l - outer * stride;
    const long long start = (long long)outer * width * stride + inner;
    for (int c = threadIdx.y; c < width; c += blockDim.y)
        result[start + (long long)c * stride] = one(sh, P, tx, cost_t + (long long)c * width, width, c, 0);
}
extern "C" __global__ void v2(const float* __restrict__ field, const float* __restrict__ cost_t, const int width,
                              const int stride, const int numel, float* __restrict__ result) {
    extern __shared__ float sh[];
    const int TX = blockDim.x, P = TX + 1, line0 = blockIdx.x * TX;
    float* row = sh + width * P;
    load_tile(field, sh, width, stride, numel, line0, TX, P);
    for (int k = threadIdx.y * TX + threadIdx.x; k < width; k += TX * blockDim.y) row[k] = cost_t[k];
    __syncthreads();
    const int tx = threadIdx.x, l = line0 + tx;
    if (l >= numel / width) return;
    const int outer = l / stride, inner = l - outer * stride;
    const long long start = (long long)outer * width * stride + inner;
    for (int c = threadIdx.y; c < width; c += blockDim.y)
        result[start + (long long)c * stride] = one(sh, P, tx, row, width, c, 1);
}
extern "C" __global__ void v3(const float* __restrict__ field, const float* __restrict__ cost_t, const int width,
                              const int stride, const int numel, float* __restrict__ result) {
    extern __shared__ float sh[];
    const int TX = blockDim.x, P = TX + 1, line0 = blockIdx.x * TX;
    float* row = sh + width * P;
    load_tile(field, sh, width, stride, numel, line0, TX, P);
    for (int k = threadIdx.y * TX + threadIdx.x; k < width; k += TX * blockDim.y) row[k] = cost_t[k];
    __syncthreads();
    const int tx = threadIdx.x, l = line0 + tx;
    if (l >= numel / width) return;
    const int outer = l / stride, inner = l - outer * stride;
    const long long start = (long long)outer * width * stride + inner;
    const int step = blockDim.y;
    for (int c0 = threadIdx.y; c0 < width; c0 += 2 * step) {
        const int c1 = c0 + step;
        if (c1 >= width) { result[start + (long long)c0 * stride] = one(sh, P, tx, row, width, c0, 1); break; }
        float m0 = NEG_INF, m1 = NEG_INF;
        bool bad = false;
        for (int j = 0; j < width; ++j) {
            const float t = sh[j * P + tx];
            const float a = __fsub_rn(t, row[abs(c0 - j)]);
            const float b = __fsub_rn(t, row[abs(c1 - j)]);
            if (isnan(a) || isnan(b)) { bad = true; break; }
            m0 = fmaxf(m0, a); m1 = fmaxf(m1, b);
        }
        if (bad || !isfinite(m0) || !isfinite(m1)) {
            result[start + (long long)c0 * stride] = one(sh, P, tx, row, width, c0, 1);
            result[start + (long long)c1 * stride] = one(sh, P, tx, row, width, c1, 1);
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
mod = cp.RawModule(code=SRC, options=("-std=c++14",))
K = {k: mod.get_function(k) for k in ("v1", "v2", "v3")}


def warp_axis(field, cost_t, width, stride):
    out = torch.empty_like(field)
    wp.launch(_grid_logsumexp_axis, dim=field.numel(), inputs=[wp.from_torch(field), wp.from_torch(cost_t), width, stride],
              outputs=[wp.from_torch(out)], device=wp.get_device("cuda:0"), stream=wp.stream_from_torch(field.device))
    return out


def make(name, TX=32, TY=4):
    ker = K[name]

    def axis(field, cost_t, width, stride):
        out = torch.empty_like(field)
        lines = field.numel() // width
        shm = (width * (TX + 1) + width) * 4
        with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
            ker(((lines + TX - 1) // TX,), (TX, TY), (cp.from_dlpack(field), cp.from_dlpack(cost_t.contiguous()),
                cp.int32(width), cp.int32(stride), cp.int32(field.numel()), cp.from_dlpack(out)), shared_mem=shm)
        return out
    return axis


def transform(fn, dual, temp):
    field = dual.reshape(-1)
    stride = field.numel()
    for width, cost in zip(dims, costs):
        stride //= width
        field = fn(field, cost / temp, width, stride)
    return field


# the Toeplitz property the v2 / v3 kernels read: cost_t[c, j] == cost_t[0, |c - j|] bit for bit
for cost in costs:
    for tv in (dx * dx, dx * dx * 7.3, 1.234e-5):
        ct = cost / torch.tensor([tv], device=dev)
        m = ct.shape[0]
        ii = torch.arange(m, device=dev)
        toe = ct[0][(ii[:, None] - ii[None, :]).abs()]
        assert torch.equal(ct.view(torch.int32), toe.view(torch.int32)), "the axis cost is not Toeplitz bit for bit"
print("Toeplitz bit for bit: yes", flush=True)
arms = [("warp", warp_axis)] + [(f"{k} TY{ty}", make(k, 32, ty)) for k in ("v1", "v2", "v3") for ty in (4, 8)]
torch.manual_seed(0)
bad = {}
for trial in range(5):
    temp = torch.tensor([dx * dx * (4.0 ** trial)], device=dev)
    d = torch.randn(n ** 3, device=dev) * 3.0
    if trial >= 1:
        d[torch.rand(n ** 3, device=dev) < 0.3] = -float("inf")
    if trial >= 2:
        d.view(n, n, n)[:, 3, :] = -float("inf")
    if trial == 3:
        d[5] = float("nan")
        d[n * n * 7 + 11] = float("nan")
    ref = transform(warp_axis, d, temp).view(torch.int32)
    for name, fn in arms[1:]:
        same = torch.equal(ref, transform(fn, d, temp).view(torch.int32))
        bad[name] = bad.get(name, 0) + (not same)
print("bit-for-bit failures by arm (5 trials):", bad, flush=True)
temp = torch.tensor([dx * dx], device=dev)
for name, fn in arms:
    d = torch.randn(n ** 3, device=dev)
    for _ in range(5):
        transform(fn, d, temp)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        transform(fn, d, temp)
    torch.cuda.synchronize()
    print(f"{name:<10s}: {(time.perf_counter() - t0) / reps * 1e3:.3f} ms a transform ({n}^3)", flush=True)
