"""Server-only CUDA gate for the accepted-step telemetry transfer boundary."""
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from physmorph.pipeline.render_reporting import accepted_render_step
from test_render_reporting import legacy_accepted_render_step, reporting_case


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='hyde06 CUDA checks')


class ScalarTransferAudit(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.host_copies = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func == torch.ops.aten._local_scalar_dense.default and args[0].is_cuda:
            raise AssertionError('Per-scalar CUDA extraction in telemetry')
        if (func == torch.ops.aten._to_copy.default and args[0].is_cuda
                and torch.device(kwargs.get('device', args[0].device)).type == 'cpu'):
            self.host_copies.append((tuple(args[0].shape), args[0].dtype))
        return func(*args, **kwargs)


@pytest.mark.parametrize('available,missing', [(True, False), (True, True), (False, True)])
def test_accepted_reporting_has_one_batched_host_transfer_on_side_stream(available, missing):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        case = reporting_case(device='cuda')
        if not available:
            case['physics'] = None
        if missing:
            case['loss_before'] = case['loss_after'] = None
        expected = legacy_accepted_render_step(**case)
        with ScalarTransferAudit() as audit:
            actual = accepted_render_step(**case)
    assert actual == expected
    count = (5 if available else 1)*len(case['leaves'])+(0 if missing else 2)+1
    assert audit.host_copies == [((count,), torch.float64)]
    assert all(t.grad is None for t in case['leaves'])


def test_cpu_loss_with_cuda_directions_still_uses_one_host_transfer():
    case = reporting_case(device='cuda')
    case['loss_before'] = case['loss_before'].cpu()
    case['loss_after'] = case['loss_after'].cpu()
    expected = legacy_accepted_render_step(**case)
    with ScalarTransferAudit() as audit:
        actual = accepted_render_step(**case)
    assert actual == expected
    assert audit.host_copies == [((18,), torch.float64)]
