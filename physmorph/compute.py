"""Explicit NumPy/CUDA execution boundary for nondifferentiable pipeline state.

Torch owns differentiable leaves; CuPy owns device array bookkeeping. No library
is monkey-patched. Participating modules import ``array_api`` explicitly. CPU
mode remains the reference implementation; CUDA mode never falls back to it.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
import importlib

import numpy as _numpy
import torch

_context = ContextVar('physmorph_compute', default=None)


def is_cuda_execution():
    return _context.get() is not None


def cuda_module():
    cp = importlib.import_module('cupy')
    if int(cp.__version__.split('.')[0]) < 14:
        raise RuntimeError('CUDA pipeline requires CuPy >=14 with its device KDTree')
    return cp


def to_array(value, dtype=None, *, copy=False):
    """Torch/Warp -> current array backend; copies are explicit ownership boundaries."""
    if hasattr(value, 'device') and type(value).__module__.startswith('warp'):
        import warp as wp
        value = wp.to_torch(value)
    if type(value).__module__.startswith('open3d') and hasattr(value, 'numpy'):
        if is_cuda_execution():
            raise RuntimeError('Open3D reconstruction belongs to the prepared input stage')
        value = value.numpy()
    if is_cuda_execution():
        cp = cuda_module()
        if torch.is_tensor(value):
            if not value.is_cuda:
                value = value.detach().to(_context.get()['device'])
            value = cp.from_dlpack(value.detach())
        result = cp.asarray(value, dtype=dtype)
    else:
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        result = _numpy.asarray(value, dtype=dtype)
    return result.copy() if copy else result


def to_host(value, *, copy_host=True):
    """Declared archive/viewer/image I/O boundary, never used for numerical work."""
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().copy()
    if hasattr(value, '__cuda_array_interface__'):
        return cuda_module().asnumpy(value)
    if isinstance(value, dict):
        return {k: to_host(v, copy_host=copy_host) for k, v in value.items()}
    if isinstance(value, list):
        return [to_host(v, copy_host=copy_host) for v in value]
    if isinstance(value, tuple):
        return tuple(to_host(v, copy_host=copy_host) for v in value)
    return value.copy() if copy_host and isinstance(value, _numpy.ndarray) else value


def warp_array(value, dtype, device, requires_grad=False):
    """An owned Warp buffer; CUDA inputs never pass through NumPy or host memory."""
    import warp as wp
    if is_cuda_execution():
        tensor = torch.as_tensor(to_array(value), device=device).contiguous()
        return wp.clone(wp.from_torch(tensor, dtype=dtype), requires_grad=requires_grad)
    return wp.array(value, dtype=dtype, device=device, requires_grad=requires_grad)


def warp_assign(destination, value):
    import warp as wp
    if is_cuda_execution():
        tensor = wp.to_torch(destination)
        tensor.copy_(torch.as_tensor(to_array(value), device=tensor.device, dtype=tensor.dtype))
    else:
        destination.assign(value)


class _ArrayAPI:
    def __getattr__(self, name):
        module = cuda_module() if is_cuda_execution() else _numpy
        function = getattr(module, name)
        if is_cuda_execution() and name in {'broadcast_to', 'clip', 'sum', 'mean', 'std',
                                           'median', 'quantile', 'percentile', 'cumsum'}:
            return lambda value, *args, **kwargs: function(self.asarray(value), *args, **kwargs)
        if is_cuda_execution() and name in {'concatenate', 'stack'}:
            return lambda values, *args, **kwargs: function([self.asarray(v) for v in values], *args, **kwargs)
        if is_cuda_execution() and name == 'repeat':
            def repeat(value, repeats, axis=None):
                tensor = torch.as_tensor(to_array(value), device=_context.get()['device'])
                count = torch.as_tensor(to_array(repeats), device=tensor.device, dtype=torch.long)
                return to_array(torch.repeat_interleave(tensor, count, dim=axis))
            return repeat
        return function

    def asarray(self, value, dtype=None, **kwargs):
        if is_cuda_execution() and isinstance(value, (list, tuple)) and value and any(
                hasattr(v, '__cuda_array_interface__') for v in value):
            value = cuda_module().stack([to_array(v) for v in value])
        if kwargs:
            module = cuda_module() if is_cuda_execution() else _numpy
            return module.asarray(to_array(value, dtype), **kwargs)
        return to_array(value, dtype)

    def ascontiguousarray(self, value, dtype=None):
        module = cuda_module() if is_cuda_execution() else _numpy
        return module.ascontiguousarray(to_array(value, dtype))


array_api = _ArrayAPI()


@lru_cache(maxsize=64)
def _input_sample_indices(n, count, seed):
    # Immutable, seeded input metadata; prepare before entering CUDA execution.
    return _numpy.random.default_rng(seed).choice(n, count, replace=False)


def sample_indices(n, count, seed=0):
    key = (int(n), int(count), int(seed))
    if is_cuda_execution():
        try:
            return _context.get()['sample_indices'][key]
        except KeyError as error:
            raise RuntimeError(f'Unprepared input sample indices: {key}') from error
    return _input_sample_indices(*key)


@lru_cache(maxsize=None)
def _execution_stream(index):
    import warp as wp
    # Reuse the per-device stream: both allocators cache blocks by stream.
    return wp.Stream(f'cuda:{index}')


@contextmanager
def cuda_execution(device, *, input_sizes=(), target_reference=None):
    dev = torch.device(device)
    if dev.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('CUDA execution requires an available CUDA device')
    cp = cuda_module()
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    prepared = {(int(n), min(int(n), 20000), 0): _input_sample_indices(int(n), min(int(n), 20000), 0)
                for n in input_sizes}
    import warp as wp
    previous = torch.cuda.current_stream(index)
    warp_stream = _execution_stream(index)
    stream = wp.stream_to_torch(warp_stream)
    stream.wait_stream(previous)
    # Warp owns the stream so its CUDA graph registry can capture/replay it.
    with torch.cuda.device(index), torch.cuda.stream(stream), cp.cuda.Device(index), \
            cp.cuda.ExternalStream(stream.cuda_stream), wp.ScopedStream(warp_stream):
        state = {'device': f'cuda:{index}',
                 'sample_indices': {key: cp.asarray(value) for key, value in prepared.items()},
                 'target_reference': target_reference}
        token = _context.set(state)
        try:
            yield
        finally:
            try:
                previous.wait_stream(stream)
            finally:
                _context.reset(token)


def prepared_target_reference():
    ref = _context.get()['target_reference'] if is_cuda_execution() else None
    if ref is None:
        raise RuntimeError('Denoised CUDA shading requires a prepared target reference asset')
    return ref


class KDTree:
    """Exact CPU or device tree, selected by the explicit execution context."""
    def __init__(self, data, **kwargs):
        self.cuda = is_cuda_execution()
        if self.cuda:
            from cupyx.scipy.spatial import KDTree as DeviceTree
            # SciPy cKDTree evaluates float64 distances even for float32 positions.
            self.data = to_array(data, _numpy.float64)
            self.tree = DeviceTree(self.data, **kwargs) if len(self.data) else None
        else:
            from scipy.spatial import cKDTree
            self.tree = cKDTree(data, **kwargs)
            self.data = self.tree.data
        self.n = len(self.data)

    def query(self, x, k=1, *, workers=None, **kwargs):
        if not self.cuda:
            return self.tree.query(x, k=k, workers=workers, **kwargs)
        cp = cuda_module()
        k = int(k)
        if k < 1:
            raise ValueError('k must be positive')
        queries = cp.ascontiguousarray(to_array(x, _numpy.float64))
        shape = queries.shape[:-1] + (() if k == 1 else (int(k),))
        if not self.n or not queries.size:
            return cp.full(shape, cp.inf), cp.full(shape, self.n, dtype=cp.int64)
        # CuPy14 does not consistently enforce distance_upper_bound (notably
        # k>N). Apply SciPy's strict radius and padding contract explicitly.
        bound = float(kwargs.pop('distance_upper_bound', float('inf')))
        take = min(k, self.n)
        distances, indices = self.tree.query(queries, k=take, **kwargs)
        valid = (distances < bound) & (indices >= 0) & (indices < self.n)
        distances = cp.where(valid, distances, cp.inf)
        indices = cp.where(valid, indices, self.n)
        if k > self.n:
            if take == 1:
                distances, indices = distances[..., None], indices[..., None]
            pad_shape = queries.shape[:-1] + (k - take,)
            distances = cp.concatenate((distances, cp.full(pad_shape, cp.inf)), axis=-1)
            indices = cp.concatenate((indices, cp.full(pad_shape, self.n, dtype=cp.int64)), axis=-1)
        return distances, indices

    def query_ball_point(self, x, r, *, workers=None, **kwargs):
        if not self.cuda:
            return self.tree.query_ball_point(x, r, workers=workers, **kwargs)
        cp = cuda_module()
        if not kwargs.pop('return_length', False):
            raise ValueError('CUDA radius queries support return_length=True only')
        if float(kwargs.pop('eps', 0.0)) != 0.0:
            raise ValueError('CUDA radius counts require exact queries (eps=0)')
        kwargs.pop('return_sorted', None)  # ordering does not affect a count
        p = kwargs.pop('p', 2.0)
        if float(p) != 2.0:
            raise ValueError('CUDA radius counts currently support Euclidean distance only')
        if kwargs:
            raise TypeError('Unsupported CUDA radius query arguments: ' + ', '.join(kwargs))
        queries = cp.ascontiguousarray(to_array(x, _numpy.float64))
        if queries.ndim < 1 or queries.shape[-1] != self.data.shape[1]:
            raise ValueError('query points must match the tree dimension')
        shape = queries.shape[:-1]
        queries = queries.reshape(-1, self.data.shape[1])
        radii = cp.broadcast_to(to_array(r, _numpy.float64), shape).reshape(-1)
        if bool(cp.any(cp.isnan(radii) | (radii < 0))):
            raise ValueError('radius must be nonnegative and not NaN')
        counts = cp.zeros(len(queries), dtype=cp.int64)
        if not self.n or not len(queries):
            return counts.reshape(shape)

        # CuPy14's radius kernel faults on the actual 300k clouds. Its exact kNN
        # query is validated independently. Count inclusive radii, doubling k only
        # where the furthest returned neighbor is still inside. There is no fixed
        # neighbor cap; bound temporary storage even for a fully dense ball.
        active = cp.arange(len(queries), dtype=cp.int64)
        k = min(16, self.n)
        while len(active):
            unresolved = []
            rows = max(1, (1 << 20) // k)
            for start in range(0, len(active), rows):
                ids = active[start:start + rows]
                distances, _ = self.query(queries[ids], k=k, p=p)
                distances = distances.reshape(len(ids), k)
                inside = distances <= radii[ids, None]
                counts[ids] = inside.sum(axis=1)
                if k < self.n:
                    unresolved.append(ids[inside[:, -1]])
            if k == self.n:
                break
            active = cp.concatenate(unresolved)
            k = min(2 * k, self.n)
        return counts.reshape(shape)


class _NdimageAPI:
    def __getattr__(self, name):
        module = importlib.import_module('cupyx.scipy.ndimage' if is_cuda_execution()
                                        else 'scipy.ndimage')
        return getattr(module, name)


ndimage = _NdimageAPI()
