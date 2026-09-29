"""Server-only device/transfer contract for world footprint observations."""
import json

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from physmorph.render.footprint_diagnostics import summarize_world_footprints
from test_footprint_diagnostics import rotation


class ScalarTransferAudit(TorchDispatchMode):
    """Counts CUDA-to-host copies and forbids per-scalar extraction (vendored from v3-grid-gs
    tests/test_render_reporting_cuda.py, whose reporting module this branch does not carry)."""
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


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='hyde06 CUDA checks')


def assert_reports_close(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_reports_close(actual[key], expected[key])
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-14)
    else:
        assert actual == expected


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_world_footprints_use_one_reduced_host_packet_on_owned_stream(dtype):
    r = rotation(dtype)
    covariance = (r @ torch.diag(torch.tensor([.014, .006, .001], dtype=dtype)) @ r.T).repeat(32, 1, 1)
    covariance[-1, 0, 0] = -1.  # Invalid rows are counted, not masked by opacity.
    case = dict(covariance=covariance, normals=r[:, 2][None].repeat(32, 1),
                support=torch.linspace(0., 1., 32, dtype=dtype), opacity=torch.ones(32, dtype=dtype),
                populations={'active_free': torch.arange(32) % 2 == 0,
                             'empty': torch.zeros(32, dtype=torch.bool)})
    expected = summarize_world_footprints(**case)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        gpu_case = {key: ({name: mask.cuda() for name, mask in value.items()} if isinstance(value, dict)
                          else value.cuda()) for key, value in case.items()}
        originals = {key: value.clone() for key, value in gpu_case.items() if torch.is_tensor(value)}
        with ScalarTransferAudit() as audit:
            actual = summarize_world_footprints(**gpu_case)
        for key, original in originals.items():
            torch.testing.assert_close(gpu_case[key], original, rtol=0, atol=0)
    assert_reports_close(actual, expected)
    # 9 validation counters + invalid/clamp counters; 8 population counters and
    # 7 summaries x 5 scalars for each of 3 built-in plus 2 caller populations.
    assert audit.host_copies == [((11+5*(8+7*5),), torch.float64)]
    assert actual['invalid_count'] == 1
    assert actual['populations']['empty']['normal_std_wu'] is None
    json.dumps(actual, allow_nan=False)
