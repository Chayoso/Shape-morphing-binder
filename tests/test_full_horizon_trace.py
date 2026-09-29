from copy import deepcopy

import numpy as np
import pytest

from scripts.probes.full_horizon import RestTrace, array_digest


@pytest.mark.parametrize('pic', [False, True])
def test_real_cpu_observer_preserves_path_and_frame_binding(monkeypatch,tmp_path,pic):
    from physmorph.mpm.state import MPMParams
    from physmorph.pipeline import PipelineConfig,runner
    source=np.random.default_rng(27).uniform(-1.5,1.5,(160,3)).astype(np.float32)
    target=(source*[1.2,.85,1.05]+[.1,0,0]).astype(np.float32)
    cfg=PipelineConfig(T=3,iters=2,animations=2,loss_res=12,render_views=2,
        render_elevs=(0.,.5),render_res=24,device='cpu',patience=5,
        outer_render_committed=True,body_ctrl=True,body_terminal_ctrl=True,
        lambda_auto=.3,w_kin=.2,w_kin_var=.3,w_ctrl=.001,w_jvol=.5,w_box=0.,max_ls_iters=1,
        adaptive_alpha=False,alpha=1e-4,replay_calibrate=False,phys_loss='ot_pace',
        loss_units='density',ot_samples=128,ot_iters=20,render_paced=True,
        commit_pic=pic,commit_pic_objective=pic,c2f_at=.5,render_res_hi=32)
    prm=MPMParams(dx=1.,nx=32,ny=32,nz=32)
    baseline=runner.run_pipeline(source,target,prm,deepcopy(cfg),log=lambda *_:None)
    trace=RestTrace(tmp_path/'trace')
    monkeypatch.setattr(runner,'optimize_window',trace.wrap(runner.optimize_window))
    observed=runner.run_pipeline(source,target,prm,deepcopy(cfg),log=lambda *_:None)
    np.testing.assert_array_equal(observed['frames'],baseline['frames'])
    np.testing.assert_array_equal(observed['F_frames'],baseline['F_frames'])
    assert observed['guards']==baseline['guards']
    for a,b in zip(observed['history'],baseline['history']):
        for k in ('loss','lambda','d_vol','d_sil','frame_end'): assert a.get(k)==b.get(k)
    summary=trace.finish(observed)
    assert len(summary['attempts'])==2 and summary['accepted_attempts']==[0,1]
    assert len(summary['events'])==1 and summary['events'][0]['c2f_render_res']==32
    assert summary['attempts'][1]['start_frame']==3
    with np.load(trace.folder/'attempt_001.npz',allow_pickle=False) as f:
        assert f['plan'].shape==(160,3) and f['pin_before'].shape==(160,)
        assert f['start_arrived'].shape==(160,)
        assert f['optimizer_terminal_speed_squared'].shape==(160,)
        assert (f['optimizer_terminal_speed_squared']>=0).all()
        promoted=observed['frames'][summary['attempts'][1]['end_frame']]
        if pic:
            assert not np.array_equal(f['optimizer_raw_endpoint'],promoted)
        else:
            np.testing.assert_array_equal(f['optimizer_raw_endpoint'],promoted)
    observed['frames'][3]=observed['frames'][3]+.1
    with pytest.raises(ValueError,match='start mismatch'): trace.finish(observed)


def test_exact_return_and_owned_inputs_then_null_reject_held_mapping(tmp_path):
    trace=RestTrace(tmp_path/'trace')
    x=np.ones((3,3),np.float32); pins=np.array([1.,0.,0.],np.float32)
    plan=x*2;arrived=np.array([True,False,True]);v=x*.1
    returned=([x,x],[],dict(v=v),None,[],dict(accepted=0,plan_img=plan,arrived_mask=arrived,pace_r=.5))
    def original(x0,**kw):
        kw['pin_init'][:]=0  # Captured pre-call pins must survive even this artificial mutation.
        return returned
    wrapped=trace.wrap(original)
    assert wrapped(x,pin_init=pins,win_index=0) is returned
    assert wrapped(x,pin_init=pins,win_index=1) is returned
    plan[:]=99;arrived[:]=False
    with np.load(trace.folder/'attempt_000.npz',allow_pickle=False) as f:
        np.testing.assert_array_equal(f['pin_before'],[1,0,0])
        np.testing.assert_array_equal(f['plan'],x*2)
        np.testing.assert_array_equal(f['start_arrived'],[1,0,1])
    result=dict(frames=[x,x.copy(),x.copy()],history=[dict(animation=0,null_commit=1),
        dict(animation=1,null_commit=1,outer_rejected=1),dict(animation=2,held=1)],
        deliver_n=2,truncation=None,n_held=1,converged=True,pinned=None)
    summary=trace.finish(result)
    assert summary['actual_last_accepted'] is None
    assert summary['attempts'][1]['start_frame']==summary['attempts'][1]['end_frame']==1
    assert summary['actual_archive_frames']==3 and summary['deliver_n']==2 and summary['held_archive_rows']==1
    result['history'][1]['frame_end']=3  # Rejected attempted frame metadata must not imply a commit.
    assert trace.finish(result)['accepted_attempts']==[]


def test_gradient_stop_has_no_invented_plan_or_frame(tmp_path):
    trace=RestTrace(tmp_path/'trace');x=np.ones((2,3),np.float32)
    returned=([x],[],None,None,[],dict(accepted=0,grad_converged=True))
    assert trace.wrap(lambda *a,**k:returned)(x,win_index=0) is returned
    result=dict(frames=[x],history=[dict(animation=0,grad_converged=1)],deliver_n=1,
                truncation=None,n_held=0,converged=True,pinned=None)
    summary=trace.finish(result)
    assert not summary['attempts'][0]['has_plan'] and summary['actual_last_accepted'] is None
