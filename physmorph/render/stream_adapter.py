"""GPU event ordering for a legacy stream-0 CUDA extension.

Torch records the custom Function on the default stream so its backward uses
that stream too. Events order inputs/outputs without host synchronization.
record_stream protects allocations until the consumer stream has finished.
Inputs are cloned on the caller stream to obtain Torch-owned storage even when
the original tensor views Warp/CuPy memory. Camera tensors are Torch-owned by
StudioRaster. The adapter adds no host synchronization; the inherited extension
still reads its bin count on the host.
"""
import torch


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)


def _owned(value):
    if isinstance(value, torch.Tensor):
        return value.clone() if value.is_cuda else value
    if isinstance(value, tuple):
        return tuple(_owned(item) for item in value)
    if isinstance(value, list):
        return [_owned(item) for item in value]
    if isinstance(value, dict):
        return {key: _owned(item) for key, item in value.items()}
    return value


class OrderedDefaultRaster(torch.nn.Module):
    def __init__(self, raster):
        super().__init__()
        self.raster = raster
        self.raster_settings = raster.raster_settings

    def forward(self, *args, **kwargs):
        x = kwargs.get('means3D', args[0] if args else None)
        if x is None or x.device.type != 'cuda':
            raise ValueError('CUDA means3D required')
        with torch.cuda.device(x.device):
            for tensor in _tensors((args, kwargs, self.raster_settings)):
                if tensor.is_cuda and tensor.device != x.device:
                    raise ValueError('All raster inputs must use the same CUDA device')
            args, kwargs = _owned(args), _owned(kwargs)
            caller = torch.cuda.current_stream(x.device)
            default = torch.cuda.default_stream(x.device)
            if caller == default:
                return self.raster(*args, **kwargs)
            default.wait_stream(caller)
            try:
                with torch.cuda.stream(default):
                    for tensor in _tensors((args, kwargs, self.raster_settings)):
                        if tensor.is_cuda:
                            tensor.record_stream(default)
                    result = self.raster(*args, **kwargs)
            finally:
                caller.wait_stream(default)
            for tensor in _tensors(result):
                if tensor.is_cuda:
                    tensor.record_stream(caller)
            return result
