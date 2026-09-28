"""Read-only reference observer through real CPU MPM/optimizer/outer promotion."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from physmorph.mpm.state import MPMParams
from physmorph.pipeline import PipelineConfig, runner


def test_reference_observer_preserves_actual_cpu_pipeline_and_expires(monkeypatch):
    source = np.random.default_rng(27).uniform(-1.5, 1.5, (160,3)).astype(np.float32)
    target = (source*[1.2,.85,1.05]+[.1,0,0]).astype(np.float32)
    cfg = PipelineConfig(T=3,iters=2,animations=2,loss_res=12,render_views=2,
        render_elevs=(0.,.5),render_res=24,device='cpu',patience=5,
        outer_render_committed=True,body_ctrl=True,body_terminal_ctrl=True,
        lambda_auto=.3,w_kin=.2,w_kin_var=0.,w_ctrl=0.,w_box=0.,max_ls_iters=1,
        adaptive_alpha=False,alpha=1e-4,replay_calibrate=False,phys_loss='ot_pace',
        loss_units='density',ot_samples=128,ot_iters=20,render_paced=True)
    prm = MPMParams(dx=1.,nx=32,ny=32,nz=32)
    baseline = runner.run_pipeline(source,target,prm,deepcopy(cfg),log=lambda *_:None)
    original = runner.optimize_window
    captured = []
    closures = []

    def observe(index, packet, common):
        terms = packet['reference'].terms(packet['xT'])
        shared = common(packet['xT'])
        merit = float(terms['volume']+packet['lambda_render']*terms['render'])+shared['value']
        assert merit == pytest.approx(packet['expected']['merit'],rel=3e-6,abs=1e-8)
        assert bool(torch.isfinite(shared['gradient']).all())
        # Mutation of an owned observer copy must never modify production state.
        packet['positions'].fill_(123.)
        packet['pins'].fill_(True)
        captured.append(index)
        closures.append((common,packet['xT']))

    def wrap(*args,**kwargs):
        result = original(*args,on_reference=observe,**kwargs)
        with pytest.raises(RuntimeError,match='expired'):
            closures[-1][0](closures[-1][1])
        return result

    monkeypatch.setattr(runner,'optimize_window',wrap)
    observed = runner.run_pipeline(source,target,prm,deepcopy(cfg),log=lambda *_:None)
    assert captured == [0,1]
    np.testing.assert_array_equal(observed['frames'],baseline['frames'])
    np.testing.assert_array_equal(observed['F_frames'],baseline['F_frames'])
    assert observed['guards'] == baseline['guards']
    for a,b in zip(observed['history'],baseline['history']):
        for key in ('loss','lambda','d_vol','d_sil','frame_end'):
            assert a.get(key) == b.get(key)
