"""Explicit hyde06 actual search/selection; N27,T20,dt.002,dx.5,grid16^3."""
import os

import pytest
import torch
import warp as wp

from physmorph.compute import cuda_execution, cuda_module
from test_post_assimilation_window_cuda import cpu_fixture, gpu_case
from test_post_selection_real import exercise_selection
from test_withdrawal_cuda import NoDeviceToHostArray

pytestmark = pytest.mark.skipif(
    os.environ.get('PHYSMORPH_POST_SELECTION_CUDA_TEST') != '1' or not torch.cuda.is_available(),
    reason='Explicit hyde06 post-assimilation selection gate')


def test_actual_gpu_search_confirmed_state_stays_on_device(monkeypatch):
    fixture = cpu_fixture(fp64=True)
    with cuda_execution('cuda:0'):
        owner, successor, cfg = gpu_case(fixture)
        def no_host(*args, **kwargs):
            pytest.fail('Numerical search/selection downloaded an array')
        with monkeypatch.context() as patch, NoDeviceToHostArray():
            patch.setattr(cuda_module(), 'asnumpy', no_host)
            patch.setattr(wp.array, 'numpy', no_host)
            report = exercise_selection(owner, successor, cfg)
        print(report)
