"""Opt-in hyde06 gate; never enabled by ordinary local test collection."""
import os

import pytest
import torch

from physmorph.compute import is_cuda_execution
from test_checkpoint_merit_terms import fixture, run_observed, check_control_derivative


pytestmark = pytest.mark.skipif(
    os.environ.get('PHYSMORPH_CUDA_TESTS') != '1' or not torch.cuda.is_available(),
    reason='Requires explicit PHYSMORPH_CUDA_TESTS=1 on hyde06')


@pytest.mark.parametrize('mode', [0, 1], ids=['displacement', 'terminal'])
def test_captured_complete_head_merit_fd_and_callback_lease(monkeypatch, mode):
    _, _, _, cfg = fixture()
    cfg.device = 'cuda:0'
    cfg.compute_backend = 'cuda'
    retained = []
    def check(packet):
        # run_pipeline owns the aligned numerical context; no fallback CPU
        # pipeline or manual default-stream switch is used by this observer.
        assert is_cuda_execution() and packet['positions'].is_cuda
        retained.append(check_control_derivative(packet, mode, capture=True))
    _, packets = run_observed(monkeypatch, check, cfg=cfg)
    assert packets[0]['optimizer_state_after_callback_exact']
    evaluate, terms, energy = retained[0]
    with pytest.raises(RuntimeError, match='expired'):
        evaluate.terms({})
    for key in ('merit', 'body_energy'):
        with pytest.raises(RuntimeError, match='gradient expired'):
            torch.autograd.grad(terms[key], energy, retain_graph=True)
    # Expiring a returned observation does not poison the caller's own leaf.
    assert torch.autograd.grad(energy*2, energy)[0] == 2
