"""Server-only CUDA scalar-transfer gate; no MPM simulation or timing claim."""
import pytest
import torch

from physmorph.pipeline.gradient_reporting import raw_direction_observations
from test_gradient_reporting import gradient_case, legacy_raw_observations
from test_render_reporting_cuda import ScalarTransferAudit


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='hyde06 CUDA checks')


@pytest.mark.parametrize('dtypes', [
    (torch.float32,)*3, (torch.float64,)*3,
    (torch.float16, torch.bfloat16, torch.float64)])
def test_raw_observations_have_one_host_copy_and_exact_side_stream_parity(dtypes):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        physics, render = gradient_case(dtypes, 'cuda')
        expected = legacy_raw_observations(physics, render)
        with ScalarTransferAudit() as audit:
            actual = raw_direction_observations(physics, render)
    assert actual == expected
    expected_dtype = torch.float64 if torch.float64 in dtypes else torch.float32
    assert audit.host_copies == [((3,), expected_dtype)]
    assert all(g.grad is None for g in physics+render)


def test_cuda_order_sensitive_mixed_dtype_accumulation_is_unchanged():
    physics = [torch.tensor([1e4], device='cuda'), torch.tensor([1.], device='cuda'),
               torch.tensor([1.], dtype=torch.float64, device='cuda')]
    render = [torch.ones_like(g) for g in physics]
    expected = legacy_raw_observations(physics, render)
    with ScalarTransferAudit() as audit:
        actual = raw_direction_observations(physics, render)
    assert actual == expected
    assert actual[0] != float(torch.sqrt(sum(g.double().pow(2).sum() for g in physics)))
    assert audit.host_copies == [((3,), torch.float64)]
