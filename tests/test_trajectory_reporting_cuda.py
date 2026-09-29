"""Server-only final health packet/stream gate; no simulation or timing claim."""
import pytest
import torch

from physmorph.pipeline.trajectory_reporting import trajectory_health
from test_render_reporting_cuda import ScalarTransferAudit
from test_trajectory_reporting import reporting_states, legacy_trajectory_health, assert_same


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='hyde06 CUDA checks')


@pytest.mark.parametrize('dtype,nonfinite', [(torch.float32, False), (torch.float64, False),
                                           (torch.float32, True), (torch.float64, True)])
def test_final_health_uses_one_host_packet_on_side_stream(dtype, nonfinite):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        states = reporting_states('cuda', dtype)
        if nonfinite:
            with torch.no_grad():
                states[0][0] = float('nan')
                states[1][0, 0, 0] = float('inf')
        before = [state.detach().clone() for state in states]
        expected = legacy_trajectory_health(states)
        with ScalarTransferAudit() as audit:
            actual = trajectory_health(iter(states))
        for state, old in zip(states, before):
            torch.testing.assert_close(state, old, rtol=0, atol=0, equal_nan=True)
            assert state.grad is None
    assert_same(actual, expected)
    assert audit.host_copies == [((len(states)+1,), torch.float64)]
