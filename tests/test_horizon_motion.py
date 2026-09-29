import json
from pathlib import Path

import numpy as np
import pytest

from scripts.probes.horizon_motion import FrameReader, window_motion, analyze, reversal_pairs, followup_statistics
from scripts.probes.coverage_paths import sha
from scripts.probes.full_horizon import array_digest


@pytest.mark.parametrize('compressed', [False, True])
def test_bounded_reader_seeks_exact_windows_and_rejects_invalid_interval(tmp_path, compressed):
    frames=np.arange(7*4*3,dtype=np.float32).reshape(7,4,3)
    path=tmp_path/'raw.npz'
    (np.savez_compressed if compressed else np.savez)(path,frames=frames)
    reader=FrameReader(path)
    try:
        np.testing.assert_array_equal(reader.read(3,5),frames[3:6])
        np.testing.assert_array_equal(reader.read(0,0),frames[:1])
        with pytest.raises(ValueError,match='outside'): reader.read(6,7)
    finally:
        reader.close()


def test_pic_correction_is_last_step_not_extra_time_and_pins_cannot_move():
    x=np.zeros((3,2,3),np.float32)
    x[1,0,0]=.2;x[2,0,0]=.1
    raw=x[-1].copy();raw[0,0]=.4
    plan=x[0].copy();pins=np.array([False,True]);arr=np.ones(2,bool)
    result=window_motion(x,raw,plan,.25,arr,pins,np.array([9.,0.]),.1,1.)
    np.testing.assert_allclose(result['path_wu'],[.3,0])
    np.testing.assert_allclose(result['raw_terminal_geometric_speed_wu_s'],[2,0])
    np.testing.assert_allclose(result['promoted_terminal_geometric_speed_wu_s'],[1,0])
    np.testing.assert_allclose(result['optimizer_terminal_stored_speed_wu_s'],[3,0])
    np.testing.assert_allclose(result['endpoint_correction_wu'],[.3,0])
    assert result['adjacent_step_reversed_count'].tolist()==[1,0]
    assert result['adjacent_step_eligible_count'].tolist()==[1,0]
    assert result['correction_changes_arrival'].tolist()==[True,False]
    assert result['correction_opposes_raw_step'].tolist()==[True,False]
    x[1,1,0]=.1
    with pytest.raises(ValueError,match='Pinned material'):
        window_motion(x,raw,plan,.25,arr,pins,np.array([9.,0.]),.1,1.)


def test_reversal_eligibility_excludes_roundoff_and_zero_pairs():
    a=np.array([[1.,0,0],[1e-6,0,0],[0,0,0]])
    eligible,reversed_=reversal_pairs(a,-a,1.)
    assert eligible.tolist()==reversed_.tolist()==[True,False,False]


def test_followup_speed_uses_observed_denominators_not_cumulative_sums():
    fields=dict(observed_windows=np.array([1.,3.,0.]),observed_steps=np.array([2.,6.,0.]),
                path_wu_sum=np.array([2.,6.,0.]),step_squared_wu2_sum=np.array([2.,6.,0.]),
                terminal_stored_speed_squared_sum=np.array([4.,12.,0.]))
    report=followup_statistics(fields)
    assert report['observed_particle_steps']==8 and report['observed_particle_windows']==4
    assert report['pooled_step_rms_wu']==1.
    assert report['pooled_optimizer_terminal_stored_speed_rms_wu_s']==2.
    assert report['per_id_normalized']['optimizer_terminal_stored_speed_rms_wu_s']['rms']==2.
    assert report['per_id_normalized']['particles']==2
    empty=followup_statistics({k:np.zeros(3) for k in fields})
    assert empty['pooled_step_rms_wu'] is None
    assert empty['pooled_optimizer_terminal_stored_speed_rms_wu_s'] is None


def make_archive(tmp_path):
    prefix=tmp_path/'full'
    source=np.array([[0,0,0],[0,2,0],[0,4,0],[0,6,0]],np.float32)
    path0=np.stack([source.copy() for _ in range(3)])
    path0[1,:,0]+=[.2,.1,0,.1]
    path0[2,:,0]+=[.1,.2,0,.2]
    raw0=path0[2].copy();raw0[0,0]=.4
    path1=np.stack([path0[-1].copy() for _ in range(3)])
    path1[1,:,0]+=[.1,0,.1,.1]
    path1[2,:,0]+=[.2,0,.2,.2]
    frames=np.concatenate([path0,path0[-1:],path1[1:],path1[-1:]])
    pins=np.array([False,True,False,False]);pinat=np.array([-1,1,-1,-1],np.int32)
    rawpath=prefix.with_name(prefix.name+'_render_full_dt_iso_nn.npz')
    np.savez(rawpath,frames=frames,src=source,pinned=pins,pinned_at=pinat,deliver_n=3)
    plan0=source.copy();plan0[:,0]+=[0,.2,1,1]
    plan1=path1[0].copy();plan1[:,0]+=[1,0,.2,1]
    folder=prefix.with_name(prefix.name+'_cohorts');folder.mkdir()
    attempts=[]
    ranges=[(0,2,True),(2,3,False),(3,3,False),(3,5,True),(5,5,False)]
    for a,(start,end,committed) in enumerate(ranges):
        plan=plan0 if a==0 else plan1
        # Rejected/null references are deliberately unrelated to accepted plans.
        if a in (1,2): plan=np.full_like(plan,100.)
        saved=dict(pin_before=pins & (pinat<=a),optimizer_raw_endpoint=raw0 if a==0 else frames[end])
        if a!=4:
            saved.update(plan=plan,radius=.25,start_arrived=np.linalg.norm(frames[start]-plan,axis=1)<=.25,
                         optimizer_terminal_speed_squared=np.arange(4,dtype=np.float64))
        path=folder/f'attempt_{a:03d}.npz';np.savez_compressed(path,**saved)
        attempts.append(dict(animation=a,sidecar=path.name,sha256=sha(path),start_frame=start,end_frame=end,
                             x0_sha256=array_digest(frames[start]),has_plan=a!=4,committed=committed))
    cfg=dict(T=2,iters=8,loss_res=36,archive_stride=1)
    history=[dict(animation=0,accepted=8,frame_end=3,d_vol=1.),
             dict(animation=1,accepted=0,null_commit=1),
             dict(animation=2,accepted=8,outer_rejected=1,frame_end=6),
             dict(animation=3,accepted=8,frame_end=6,d_vol=.7),
             dict(animation=4,accepted=0,grad_converged=1),dict(animation=5,held=1)]
    record=dict(config=cfg,mpm=dict(dt=.1,dx=.3),history=history,guards={},
                arms=dict(render_full_dt_iso_nn=dict(deliver_n=3)))
    recordpath=prefix.with_suffix('.json');recordpath.write_text(json.dumps(record))
    root=Path(__file__).resolve().parents[1]
    code={str(root/k):sha(root/k) for k in ('physmorph/compute.py','physmorph/pipeline/settlement.py')}
    protocolpath=prefix.with_suffix('.protocol.json');protocolpath.write_text(json.dumps(dict(arm='baseline',code=code)))
    renderpath=prefix.with_name(prefix.name+'.render_influence.json')
    renderpath.write_text(json.dumps(dict(causal_render_ablation='not measured')))
    trace=dict(attempts=attempts,inputs_code_unchanged=True,actual_archive_frames=len(frames),
               actual_last_accepted=3,accepted_attempts=[0,3],reported_converged=True,deliver_n=3,held_archive_rows=1,truncation={},
               protocol_sha256=sha(protocolpath),result_sha256=sha(recordpath),
               output_sha256={str(p):sha(p) for p in (rawpath,renderpath)})
    prefix.with_suffix('.rest_trace.json').write_text(json.dumps(trace))
    return prefix,frames


def test_archive_analysis_separates_accepted_relabels_pins_and_delivery(tmp_path):
    prefix,frames=make_archive(tmp_path)
    report,data=analyze(prefix)
    assert report['physical_accepted_windows']==2
    assert [r['animation'] for r in report['windows']]==[0,3]
    assert [r['animation'] for r in report['reference_relabels']]==[3]
    assert report['reference_relabels'][0]['newly_labelled']==1
    assert report['reference_relabels'][0]['no_longer_labelled']==1
    assert report['delivery_scope']['animation']==0
    assert report['delivery_frame']==2 and report['actual_frames']==7
    assert report['held_archive_rows']==1
    assert data['first_accepted_endpoint_arrival'].tolist()==[0,0,3,-1]
    assert data['after_endpoint_arrival_free_observed_windows'].tolist()==[1,0,0,0]
    assert data['after_endpoint_arrival_pinned_observed_windows'].tolist()==[0,1,0,0]
    assert data['after_endpoint_arrival_free_observed_steps'].tolist()==[2,0,0,0]
    np.testing.assert_allclose(data['after_endpoint_arrival_free_path_wu_sum'],[.2,0,0,0])
    assert data['after_endpoint_arrival_pinned_path_wu_sum'].tolist()==[0,0,0,0]
    # Saved PIC last displacement is negative; next physical displacement is positive.
    assert data['boundary_step_reversed_count'].tolist()==[1,0,0,0]
    assert data['reference_no_longer_labelled_count'].tolist()==[1,0,0,0]
    assert report['windows'][1]['cohorts']['previously_arrived_still_free']['particles']==1
    assert report['render_influence']['causal_render_ablation']=='not measured'
    # Detect mutation of a bound raw archive even if a valid NPZ replaces it.
    np.savez(prefix.with_name(prefix.name+'_render_full_dt_iso_nn.npz'),frames=frames)
    with pytest.raises(ValueError,match='output hash mismatch'): analyze(prefix)


def test_analysis_rejects_false_commit_and_changed_producer_dependency(tmp_path):
    prefix,_=make_archive(tmp_path)
    tracepath=prefix.with_suffix('.rest_trace.json')
    trace=json.loads(tracepath.read_text());trace['attempts'][2]['committed']=True
    tracepath.write_text(json.dumps(trace))
    with pytest.raises(ValueError,match='commit mismatch'): analyze(prefix)
    trace['attempts'][2]['committed']=False
    protocolpath=prefix.with_suffix('.protocol.json')
    protocol=json.loads(protocolpath.read_text())
    protocol['code'][next(iter(protocol['code']))]='0'*64
    protocolpath.write_text(json.dumps(protocol))
    trace['protocol_sha256']=sha(protocolpath);tracepath.write_text(json.dumps(trace))
    with pytest.raises(ValueError,match='Producer dependency mismatch'): analyze(prefix)


def test_analysis_rejects_delivery_and_attempt_metadata_substitution(tmp_path):
    prefix,_=make_archive(tmp_path)
    path=prefix.with_suffix('.rest_trace.json')
    trace=json.loads(path.read_text());trace['deliver_n']=6
    path.write_text(json.dumps(trace))
    with pytest.raises(ValueError,match='Delivery metadata mismatch'): analyze(prefix)
    trace['deliver_n']=3;trace['attempts'][1]['animation']=0
    path.write_text(json.dumps(trace))
    with pytest.raises(ValueError,match='trace attempt'): analyze(prefix)
